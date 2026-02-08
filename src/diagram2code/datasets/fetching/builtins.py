from __future__ import annotations

from .descriptors import Artifact, DatasetDescriptor


def built_in_descriptors() -> dict[str, DatasetDescriptor]:
    # Pinned to a specific commit for reproducibility.
    revision = "35d7dc891fd0ac17f3773aeeba023fe15acbd062"

    return {
        "flowlearn": DatasetDescriptor(
            name="flowlearn",
            version=f"hf-{revision[:7]}",
            description="FlowLearn dataset snapshot from Hugging Face (pinned revision).",
            homepage="https://huggingface.co/datasets/jopan/FlowLearn",
            artifacts=(
                Artifact(
                    url=f"hf://datasets/jopan/FlowLearn@{revision}",
                    sha256=revision,  # store the pinned revision here
                    type="hf_snapshot",
                    target_subdir="raw",
                ),
            ),
            expected_layout=(),  # normalization comes next step (prepared/)
            loader_hint="flowlearn_hf_snapshot_v1",
        ),
    }
