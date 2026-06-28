# jlc_controlnet_aux_preproc_lists.py

"""
Curated include/exclude policy for the JLC Aux Preprocessor Wrapper.

The dynamic wrapper sits on top of Fannovel16's comfyui_controlnet_aux
package, but intentionally exposes only preprocessors that fit a simple
shared-widget model:

    IMAGE input
    optional/shared resolution input
    IMAGE-only output

Preprocessors that require thresholds, detector toggles, model selectors,
pose/keypoint outputs, masks, or custom payloads should be excluded from this
wrapper and used through their native ControlNet Auxiliary nodes instead.
"""

# -------------------------------------------------------------------
# Autodiscovery
# -------------------------------------------------------------------
# True means the wrapper may include currently installed aux preprocessors
# that pass the strict structural gate, even if they are not named below.
# This prevents the local list from becoming stale as Fannovel's package grows,
# while still keeping the node simple.
# -------------------------------------------------------------------
JLC_CNAUX_AUTODISCOVER_SIMPLE_PREPROCESSORS = True


# -------------------------------------------------------------------
# Blacklist
# -------------------------------------------------------------------
# Names must match Fannovel's exported AUX_NODE_MAPPINGS keys.
# This list catches known non-simple families before structural inspection.
# -------------------------------------------------------------------
JLC_CNAUX_PREPROCESSOR_BLACKLIST = [
    # Wrapper passthrough / non-image or special-case behavior
    "InpaintPreprocessor",
    "DiffusionEdge_Preprocessor",

    # Parameter-heavy edge/line preprocessors. Use native nodes so thresholds
    # and mode widgets remain visible and intentional.
    "CannyEdgePreprocessor",
    "PyraCannyPreprocessor",
    "M-LSDPreprocessor",
    "LineArtPreprocessor",
    "LineartStandardPreprocessor",
    "ScribblePreprocessor",
    "Scribble_XDoG_Preprocessor",
    "Scribble_PiDiNet_Preprocessor",
    "FakeScribblePreprocessor",
    "PiDiNetPreprocessor",

    # Pose/keypoint nodes with detector toggles, model selectors, JSON/keypoint
    # outputs, or other non-simple behavior.
    "OpenposePreprocessor",
    "DWPreprocessor",
    "AnimalPosePreprocessor",
    "SavePoseKpsAsJsonFile",
    "FacialPartColoringFromPoseKps",
    "UpperBodyTrackingFromPoseKps",
    "RenderPeopleKps",
    "RenderAnimalKps",
    "DensePosePreprocessor",
    "MediaPipe-FaceMeshPreprocessor",

    # Optical flow / mask / special-output preprocessors
    "Unimatch_OptFlowPreprocessor",
    "MaskOptFlow",
    "SAMPreprocessor",

    # Segmentation families tend to be dependency-heavy or special-case.
    "UniFormer-SemSegPreprocessor",
    "UniformerPreprocessor",
    "OneFormer-COCO-SemSegPreprocessor",
    "OneFormer-ADE20K-SemSegPreprocessor",
    "AnimeFace_SemSegPreprocessor",
    "SemSegPreprocessor",

    # MeshGraphormer families are dependency-heavy / special-case.
    "MeshGraphormer-DepthMapPreprocessor",
    "MeshGraphormer+ImpactDetector-DepthMapPreprocessor",
]


# -------------------------------------------------------------------
# Preferred whitelist / ordering
# -------------------------------------------------------------------
# This list is primarily an ordering preference and compatibility fallback.
# The wrapper still verifies that each installed node is structurally simple
# before exposing it.
# -------------------------------------------------------------------
JLC_CNAUX_PREPROCESSOR_WHITELIST = [
    "DISABLED",

    # Edges / linework likely to be simple in common aux builds
    "HEDPreprocessor",
    "TEEDPreprocessor",
    "AnimeLineArtPreprocessor",
    "Manga2Anime_LineArt_Preprocessor",
    "AnyLineArtPreprocessor_aux",
    "BinaryPreprocessor",

    # Depth
    "Zoe-DepthMapPreprocessor",
    "DepthAnythingPreprocessor",
    "Zoe_DepthAnythingPreprocessor",
    "DepthAnythingV2Preprocessor",
    "MiDaS-DepthMapPreprocessor",
    "LeReS-DepthMapPreprocessor",
    "Metric3D-DepthMapPreprocessor",

    # Normals
    "DSINE-NormalMapPreprocessor",
    "BAE-NormalMapPreprocessor",
    "MiDaS-NormalMapPreprocessor",
    "Metric3D-NormalMapPreprocessor",

    # Color / utility image transforms
    "ColorPreprocessor",
    "ImageLuminanceDetector",
    "ImageIntensityDetector",
    "ShufflePreprocessor",
    "TilePreprocessor",
    "TTPlanet_TileGF_Preprocessor",
    "TTPlanet_TileSimple_Preprocessor",
]
