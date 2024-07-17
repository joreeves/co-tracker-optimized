# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Tuple

import torch
import torch.nn.functional as F

import math

from einops import rearrange

from cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid
from cotracker.models.build_cotracker import build_cotracker


class CoTrackerPredictor(torch.nn.Module):
    def __init__(self, checkpoint="./checkpoints/cotracker2.pth", batches:int=4, visibility_threshold=0.9):
        """
        Increase the batch size for larger queries. The default batch size is 4.
        """
        super().__init__()
        self.iters = 6
        self.support_grid_size = 6
        self.visibility_threshold = visibility_threshold
        self.batches = batches
        model = build_cotracker(checkpoint)
        self.interp_shape = model.model_resolution
        self.model = model
        self.model.eval()

    @torch.autocast(device_type="cuda")
    @torch.no_grad()
    def forward(
        self,
        video,  # (B, T, 3, H, W)
        # input prompt types:
        # - None. Dense tracks are computed in this case. You can adjust *query_frame* to compute tracks starting from a specific frame.
        # *backward_tracking=True* will compute tracks in both directions.
        # - queries. Queried points of shape (B, N, 3) in format (t, x, y) for frame index and pixel coordinates.
        # - grid_size. Grid of N*N points from the first frame. if segm_mask is provided, then computed only for the mask.
        # You can adjust *query_frame* and *backward_tracking* for the regular grid in the same way as for dense tracks.
        queries: torch.Tensor = None,
        segm_mask: torch.Tensor = None,  # Segmentation mask of shape (B, 1, H, W)
        grid_size: int = 0,
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        backward_tracking: bool = False,
        add_support_grid: bool = False,
    ):
        if queries is None and grid_size == 0:
            tracks, visibilities = self._compute_dense_tracks(
                video,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
            )
        else:
            tracks, visibilities = self._compute_sparse_tracks(
                video,
                queries,
                segm_mask,
                grid_size,
                add_support_grid=add_support_grid, #(grid_size == 0 or segm_mask is not None),
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
            )

        return tracks, visibilities

    def _compute_dense_tracks(self, video, grid_query_frame, grid_size=80, backward_tracking=False):
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = None
        grid_pts = torch.zeros((1, grid_width * grid_height, 3)).to(video.device)
        grid_pts[0, :, 0] = grid_query_frame
        for offset in range(grid_step * grid_step):
            print(f"step {offset} / {grid_step * grid_step}")
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[0, :, 1] = torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            grid_pts[0, :, 2] = (
                torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy
            )
            tracks_step, visibilities_step = self._compute_sparse_tracks(
                video=video,
                queries=grid_pts,
                backward_tracking=backward_tracking,
            )
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)

        return tracks, visibilities

    def _add_support_grid(self, queries):
        grid_pts = get_points_on_a_grid(
            self.support_grid_size, self.interp_shape, device=queries.device
        )
        grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
        grid_pts = grid_pts.repeat(queries.size(0), 1, 1).type(queries.dtype)
        queries = torch.cat([queries, grid_pts], dim=1)
        return queries
    
    def _remove_support_grid(self, tracks, visibilities):
        return tracks[:, :, :-self.support_grid_size**2], visibilities[:, :, :-self.support_grid_size**2]

    def _compute_tracks(self, video, queries, support_grid=False, iters=6):
        B, P, C = queries.shape

        # print(queries.shape)

        pad = self.batches - (queries.size(1) % self.batches)

        masked = torch.ones_like(queries)

        masked = F.pad(masked, (0, 0, 0, pad), value=0).bool()
        queries = F.pad(queries, (0, 0, 0, pad), value=0)

        # print(queries.shape)

        tracks, visibilities = None, None

        tracks = torch.zeros((B, video.size(1), queries.size(1), 2), dtype=queries.dtype, device=queries.device)
        visibilities = torch.zeros((B, video.size(1), queries.size(1)), dtype=queries.dtype, device=queries.device)

        for i in range(self.batches):
            queries_batch = queries[:, i::self.batches]

            if support_grid:
                queries_batch = self._add_support_grid(queries_batch)

            tracks_batch, visibilities_batch, __ = self.model.forward(video=video, queries=queries_batch, iters=self.iters)

            if support_grid:
                tracks_batch, visibilities_batch = self._remove_support_grid(tracks_batch, visibilities_batch)
                
            tracks[:, :, i::self.batches] = tracks_batch
            visibilities[:, :, i::self.batches] = visibilities_batch

            break

        return tracks[:, :, :-pad], visibilities[:, :, :-pad]
    
    def _compute_sparse_tracks(
        self,
        video,
        queries,
        segm_mask=None,
        grid_size:Union[Tuple[int, ...], int]=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
    ):
        B, T, C, H, W = video.shape

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        check_grid_size = grid_size > 0 if isinstance(grid_size, int) else (grid_size[0] > 0 and grid_size[1] > 0)
        
        if queries is not None:
            B, N, D = queries.shape
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (self.interp_shape[1] - 1) / (W - 1),
                    (self.interp_shape[0] - 1) / (H - 1),
                ]
            )
        elif check_grid_size:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)
            if segm_mask is not None:
                segm_mask = F.interpolate(segm_mask, tuple(self.interp_shape), mode="nearest")
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            ).repeat(B, 1, 1).type(video.dtype)  # Make sure dtype matches

        # if add_support_grid:
        #     queries = self._add_support_grid(queries, self.support_grid_size, self.interp_shape)

        tracks, visibilities = self._compute_tracks(
            video=video, queries=queries, support_grid=add_support_grid
            )

        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(
                video, queries, tracks, visibilities, add_support_grid
            )
            # if add_support_grid:
            #     queries[:, -self.support_grid_size**2 :, 0] = T - 1

        # if add_support_grid:  # Remove support grid points
        #     tracks, visibilities = self._remove_support_grid(tracks, visibilities)

        thr = 0.9
        visibilities = visibilities > self.visibility_threshold

        # correct query-point predictions
        # see https://github.com/facebookresearch/co-tracker/issues/28

        # TODO: batchify
        for i in range(len(queries)):
            queries_t = queries[i, : tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, : tracks.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibilities[i, queries_t, arange] = True

        tracks *= tracks.new_tensor(
            [(W - 1) / (self.interp_shape[1] - 1), (H - 1) / (self.interp_shape[0] - 1)]
        )
        return tracks, visibilities

    def _compute_backward_tracks(self, video, queries, tracks, visibilities, add_support_grid):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

        inv_tracks, inv_visibilities = self._compute_tracks(
            video=inv_video, queries=inv_queries, support_grid=add_support_grid
            )

        inv_tracks = inv_tracks.flip(1)
        inv_visibilities = inv_visibilities.flip(1)
        arange = torch.arange(video.shape[1], device=queries.device)[None, :, None]

        mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, 2)

        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        return tracks, visibilities


class CoTrackerOnlinePredictor(torch.nn.Module):
    def __init__(self, checkpoint="./checkpoints/cotracker2.pth"):
        super().__init__()
        self.support_grid_size = 6
        model = build_cotracker(checkpoint)
        self.interp_shape = model.model_resolution
        self.step = model.window_len // 2
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video_chunk,
        is_first_step: bool = False,
        queries: torch.Tensor = None,
        grid_size: int = 10,
        grid_query_frame: int = 0,
        add_support_grid=False,
    ):
        B, T, C, H, W = video_chunk.shape
        # Initialize online video processing and save queried points
        # This needs to be done before processing *each new video*
        if is_first_step:
            self.model.init_video_online_processing()
            if queries is not None:
                B, N, D = queries.shape
                assert D == 3
                queries = queries.clone()
                queries[:, :, 1:] *= queries.new_tensor(
                    [
                        (self.interp_shape[1] - 1) / (W - 1),
                        (self.interp_shape[0] - 1) / (H - 1),
                    ]
                )
            elif grid_size > 0:
                grid_pts = get_points_on_a_grid(
                    grid_size, self.interp_shape, device=video_chunk.device
                )
                queries = torch.cat(
                    [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                    dim=2,
                )
            if add_support_grid:
                grid_pts = get_points_on_a_grid(
                    self.support_grid_size, self.interp_shape, device=video_chunk.device
                )
                grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
                queries = torch.cat([queries, grid_pts], dim=1)
            self.queries = queries
            return (None, None)

        video_chunk = video_chunk.reshape(B * T, C, H, W)
        video_chunk = F.interpolate(
            video_chunk, tuple(self.interp_shape), mode="bilinear", align_corners=True
        )
        video_chunk = video_chunk.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        tracks, visibilities, __ = self.model(
            video=video_chunk,
            queries=self.queries,
            iters=self.iters,
            is_online=True,
        )
        thr = 0.9
        return (
            tracks
            * tracks.new_tensor(
                [
                    (W - 1) / (self.interp_shape[1] - 1),
                    (H - 1) / (self.interp_shape[0] - 1),
                ]
            ),
            visibilities > thr,
        )
