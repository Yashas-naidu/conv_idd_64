# Mixed Dataset Training for ConvRNN

This project now supports training on a mixed dataset combining three different sources:
- **IDD (Indian Driving Dataset)**: 50% of training data
- **Japanese Streets**: 30% of training data  
- **vKITTI**: 20% of training data

## Dataset Structure

### IDD Dataset
```
/mnt/local/gs_datasets/idd-train-set4/idd_temporal_train_4/
├── 00163/
│   ├── 155270_leftImg8bit/
│   │   ├── frame_000000.jpeg
│   │   ├── frame_000001.jpeg
│   │   └── ...
│   └── ...
└── ...
```

### Japanese Streets Dataset
```
/mnt/local/gs_datasets/japanese-streets/
└── frames/
    ├── group_0001/
    │   ├── set_0001/
    │   │   ├── frame_s000000_i00.jpg
    │   │   ├── frame_s000001_i01.jpg
    │   │   └── ...
    │   ├── set_0002/
    │   └── ...
    └── ...
```

### vKITTI Dataset
```
/mnt/local/gs_datasets/vkitti/
├── Scene01/
│   ├── 15-deg-left/
│   │   └── frames/
│   │       └── rgb/
│   │           ├── Camera_0/
│   │           │   ├── rgb_00000.jpg
│   │           │   ├── rgb_00001.jpg
│   │           │   └── ...
│   │           └── Camera_1/
│   ├── 15-deg-right/
│   ├── 30-deg-left/
│   ├── 30-deg-right/
│   ├── clone/
│   ├── fog/
│   ├── morning/
│   ├── overcast/
│   ├── rain/
│   └── sunset/
└── ...
```

## Files Added/Modified

### New Files
- `mixed_dataset.py` - Main mixed dataset implementation
- `train_mixed.py` - Training script for mixed dataset
- `test_mixed_dataset.py` - Test script to verify dataset loading
- `README_MIXED_DATASET.md` - This documentation

### Modified Files
- `main.py` - Updated to support mixed dataset (but kept for backward compatibility)
- `data.py` - Original IDD-only dataset (kept for reference)

## Usage

### 1. Test the Mixed Dataset

First, test if all datasets can be loaded correctly:

```bash
python test_mixed_dataset.py
```

This will verify that:
- All dataset paths are accessible
- Each dataset can be loaded individually
- The combined mixed dataset works correctly
- Data samples can be loaded without errors

### 2. Train with Mixed Dataset

Use the new training script:

```bash
python train_mixed.py \
    --idd_path /mnt/local/gs_datasets/idd-train-set4/idd_temporal_train_4 \
    --japanese_path /mnt/local/gs_datasets/japanese-streets \
    --vkitti_path /mnt/local/gs_datasets/vkitti \
    --batch_size 70 \
    --epochs 50 \
    --lr 1e-4
```

### 3. Customize Dataset Ratios

You can adjust the proportions of each dataset:

```bash
python train_mixed.py \
    --idd_path /path/to/idd \
    --japanese_path /path/to/japanese \
    --vkitti_path /path/to/vkitti \
    --idd_ratio 0.6 \
    --japanese_ratio 0.25 \
    --vkitti_ratio 0.15
```

### 4. Adjust Motion Detection

Control the motion threshold for filtering sequences:

```bash
python train_mixed.py \
    --idd_path /path/to/idd \
    --japanese_path /path/to/japanese \
    --vkitti_path /path/to/vkitti \
    --motion_threshold 0.15
```

## Key Features

### Motion Detection
- Automatically filters out sequences with insufficient motion
- Configurable threshold (default: 0.2)
- Speeds up training by focusing on dynamic sequences

### Balanced Sampling
- Maintains specified ratios across datasets
- Ensures diverse training data
- Prevents bias toward any single dataset

### Frame Stride
- Uses 5-frame stride between input/output frames
- 2 input frames → 1 output frame
- Reduces redundancy while maintaining temporal coherence

### Automatic Validation Split
- 80% training, 20% validation
- Each dataset split independently
- Maintains ratios in both splits

## Training Configuration

### Default Parameters
- **Input frames**: 2
- **Output frames**: 1
- **Frame stride**: 5
- **Target size**: 256x256
- **Batch size**: 70
- **Learning rate**: 1e-4
- **Epochs**: 50
- **Motion threshold**: 0.2

### Model Architecture
- **Default**: ConvGRU
- **Alternative**: ConvLSTM (use `--convlstm` flag)
- **Multi-GPU**: Automatic DataParallel if multiple GPUs available

## Output

### Model Checkpoints
- Saved to `./save_model/{timestamp}/`
- Includes model state, optimizer state, and training arguments
- Automatic early stopping with patience=20

### Training Logs
- TensorBoard logs in `./runs/{timestamp}/`
- Loss history in `avg_train_losses.txt` and `avg_valid_losses.txt`
- Real-time training progress with tqdm

### Dataset Statistics
- Console output shows dataset composition
- Validation of data loading
- Motion filtering statistics

## Troubleshooting

### Common Issues

1. **Dataset paths not found**
   - Verify all paths exist and are accessible
   - Check file permissions
   - Ensure correct directory structure

2. **Empty dataset after motion filtering**
   - Lower the motion threshold (e.g., `--motion_threshold 0.1`)
   - Check if sequences have enough frames (minimum 10 for stride=5)

3. **Memory issues**
   - Reduce batch size
   - Reduce number of workers
   - Use smaller target size

4. **Loading errors**
   - Check image file formats (.jpeg for IDD, .jpg for others)
   - Verify image files are not corrupted
   - Check file naming conventions

### Debug Mode

Run the test script first to identify issues:

```bash
python test_mixed_dataset.py
```

This will show detailed information about each dataset and help identify problems.

## Performance Tips

1. **Use SSD storage** for faster data loading
2. **Adjust num_workers** based on your system (default: 40)
3. **Enable pin_memory** for faster GPU transfer
4. **Monitor GPU memory** and adjust batch size accordingly
5. **Use mixed precision** if available (requires additional setup)

## Compatibility

- **Python**: 3.7+
- **PyTorch**: 1.8+
- **CUDA**: 10.2+ (for GPU training)
- **Dependencies**: See `requirements.txt`

The mixed dataset maintains the same interface as the original IDD dataset, so existing code should work without modification.
