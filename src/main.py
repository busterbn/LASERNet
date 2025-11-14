from src.dataset import (
    FieldType,
    LaserNetDataRepository,
    Plane,
    PlaneSelection,
    build_datasets,
    build_dataloaders,
)

def main() -> None:
    repo = LaserNetDataRepository(axis_scan_files=1, frame_cache_size=4)
    plane = PlaneSelection(Plane.XY, index=0)  # choose dominant XY slice
    datasets = build_datasets(
        repository=repo,
        sequence_length=4,
        field=FieldType.TEMPERATURE,
        plane=plane,
        target_offset=1,
        padding_value=0.0,
    )
    loaders = build_dataloaders(datasets, batch_size=2, num_workers=2)


if __name__ == "__main__":
    main()