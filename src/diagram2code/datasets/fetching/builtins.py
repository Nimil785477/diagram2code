from __future__ import annotations

from diagram2code.datasets.fetching.descriptors import Artifact, DatasetDescriptor


def built_in_descriptors() -> dict[str, DatasetDescriptor]:
    """
    Built-in remote dataset descriptors.

    Notes:
    - flowlearn is large (HF snapshot) and requires explicit confirmation (--yes).
    - tiny_remote_v1 is intentionally small and used for fetch/manifest smoke tests.
    """

    flowlearn_revision = "35d7dc891fd0ac17f3773aeeba023fe15acbd062"

    return {
        "flowlearn": DatasetDescriptor(
            name="flowlearn",
            version=f"hf-{flowlearn_revision[:7]}",
            description="FlowLearn dataset snapshot from Hugging Face (pinned revision).",
            homepage="https://huggingface.co/datasets/jopan/FlowLearn",
            artifacts=(
                Artifact(
                    url=f"hf://datasets/jopan/FlowLearn@{flowlearn_revision}",
                    sha256=flowlearn_revision,  # pinned HF revision
                    type="hf_snapshot",
                    target_subdir="raw",
                ),
            ),
            expected_layout=(),
            loader_hint="flowlearn_hf_snapshot_v1",
        ),
        "tiny_remote_v1": DatasetDescriptor(
            name="tiny_remote_v1",
            version="1",
            description="Tiny remote dataset used for fetch/manifest CLI smoke tests.",
            homepage="https://github.com/Nimil785477/diagram2code",
            artifacts=(
                Artifact(
                    # Small, stable text file (used only for fetch verification)
                    url=(
                        "https://raw.githubusercontent.com/Nimil785477/diagram2code/main/README.md"
                    ),
                    # TODO: replace with actual sha256 (see instructions)
                    sha256="c0e454f9541055c14dc1a124915d5c19be0d40bc268feff0ff401647fbfb34c6",
                    type="file",
                    target_subdir="raw",
                ),
            ),
            expected_layout=(),
            loader_hint="tiny_remote_file_v1",
        ),
    }
