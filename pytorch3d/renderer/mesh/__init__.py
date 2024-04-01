# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from .clip import (
    clip_faces,
    ClipFrustum,
    ClippedFaces,
    convert_clipped_rasterization_to_original_faces,
)

# pyre-fixme[21]: Could not find module `pytorch3d.renderer.mesh.rasterize_meshes`.
from .rasterize_meshes import rasterize_meshes
from .rasterizer import MeshRasterizer, RasterizationSettings
from .renderer import MeshRenderer, MeshRendererWithFragments
from .shader import (  # DEPRECATED
    BlendParams,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    SplatterPhongShader,
    BakerBlinnPhong,
    MultiTexturedSoftPhongShader,
    TexturedSoftPhongShader,
)
from .shading import gouraud_shading, phong_shading
from .textures import (  # DEPRECATED
    Textures,
    TexturesAtlas,
    TexturesBase,
    TexturesUV,
    TexturesUV_v06,
    TexturesVertex,
)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
