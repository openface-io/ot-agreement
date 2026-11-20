from pathlib import Path
from zipfile import ZipExtFile, ZipFile

import pandas as pd
import pytest

from ot_agreement.prepare_data.prepare_datasets import unpack_raw_annotations, write_csv


def test_write_csv_with_overlapping_writers(tmp_path, monkeypatch):
    """Publish both CSVs when a second writer finishes during the first write."""
    output = tmp_path / "prepared" / "annotations.csv"
    first = pd.DataFrame({"annotation_id": [1, 2]})
    second = pd.DataFrame({"annotation_id": [3]})
    to_csv = pd.DataFrame.to_csv

    def write_with_overlap(frame, path, **kwargs):
        assert Path(path).parent == output.parent
        to_csv(frame, path, **kwargs)
        if frame is first:
            write_csv(second, output)
            pd.testing.assert_frame_equal(pd.read_csv(output), second)

    monkeypatch.setattr(pd.DataFrame, "to_csv", write_with_overlap)

    write_csv(first, output)

    pd.testing.assert_frame_equal(pd.read_csv(output), first)
    assert list(output.parent.iterdir()) == [output]


def test_unpack_raw_annotations_can_retry_after_interruption(tmp_path, monkeypatch):
    """Discard a partially copied member so the next call can extract it."""
    archive_path = tmp_path / "raw_annotations.zip"
    output = tmp_path / "nested" / "annotation.xml"
    contents = b"<annotation/>"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/annotation.xml", contents)
    read = ZipExtFile.read

    def interrupt_after_first_chunk(source, size=-1):
        if source.tell():
            raise KeyboardInterrupt
        return read(source, 4)

    with monkeypatch.context() as interruption:
        interruption.setattr(ZipExtFile, "read", interrupt_after_first_chunk)
        with pytest.raises(KeyboardInterrupt):
            unpack_raw_annotations(tmp_path)

    assert list(output.parent.iterdir()) == []
    unpack_raw_annotations(tmp_path)
    assert output.read_bytes() == contents


def test_unpack_raw_annotations_preserves_existing_and_concurrent_files(tmp_path, monkeypatch):
    """Keep user files that exist before extraction or arrive during it."""
    archive_path = tmp_path / "raw_annotations.zip"
    existing = tmp_path / "existing.xml"
    concurrent = tmp_path / "concurrent.xml"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(existing.name, b"archived existing")
        archive.writestr(concurrent.name, b"archived concurrent")
    existing.write_bytes(b"user existing")
    open_member = ZipFile.open

    def open_with_publisher(archive, member, *args, **kwargs):
        if member.filename == concurrent.name:
            with concurrent.open("xb") as published:
                published.write(b"user concurrent")
        return open_member(archive, member, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", open_with_publisher)

    unpack_raw_annotations(tmp_path)

    assert existing.read_bytes() == b"user existing"
    assert concurrent.read_bytes() == b"user concurrent"
    assert set(tmp_path.iterdir()) == {archive_path, existing, concurrent}
