# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_qwen3_vl": ["Qwen3VLConfig", "Qwen3VLTextConfig", "Qwen3VLVisionConfig"],
    "processing_qwen3_vl": ["Qwen3VLProcessor"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modular_qwen3_vl"] = [
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLModel",
        "Qwen3VLPreTrainedModel",
        "Qwen3VLVisionModel",
    ]


if TYPE_CHECKING:
    from .configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
    from .processing_qwen3_vl import Qwen3VLProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modular_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLModel,
            Qwen3VLPreTrainedModel,
            Qwen3VLVisionModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
