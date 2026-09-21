from open_llm_vtuber.artifacts import ArtifactManager, ArtifactType


def test_artifact_has_stable_id_and_serialization():
    manager = ArtifactManager()
    artifact = manager.create(
        name="index.html",
        artifact_type=ArtifactType.FILE,
        mime_type="text/html",
        workspace_id="workspace_001",
        path="index.html",
    )

    assert artifact.artifact_id.startswith("artifact_")
    assert artifact.to_dict()["artifact_id"] == artifact.artifact_id
    assert artifact.to_dict()["artifact_type"] == "file"


def test_update_keeps_identity():
    manager = ArtifactManager()
    artifact = manager.create(name="index.html")
    original_id = artifact.artifact_id

    updated = manager.update(original_id, name="login.html", size=123)

    assert updated.artifact_id == original_id
    assert updated.name == "login.html"
    assert updated.size == 123


def test_delete_removes_only_record():
    manager = ArtifactManager()
    artifact = manager.create(name="notes.txt")

    assert manager.delete(artifact.artifact_id) is True
    assert manager.get(artifact.artifact_id) is None
    assert manager.delete(artifact.artifact_id) is False
