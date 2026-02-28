# How to Run the Project - Beginner's Guide

## Step-by-Step Instructions

### Step 1: Open Terminal

**On Mac:**
1. Press `Command + Space` (opens Spotlight search)
2. Type "Terminal" and press Enter
3. A black window will open - this is your terminal!

**Or:**
- Go to Applications → Utilities → Terminal

### Step 2: Navigate to Your Project Folder

In the terminal, type this command and press Enter:

```bash
cd "/Users/doraweng/Documents/Kaggle Project/Hedege Fund-Time series forecast"
```

**What this does:** Changes directory to your project folder.

**Tip:** You can copy-paste this command into the terminal!

### Step 3: Run the Quick Training Script

Now type this command and press Enter:

```bash
./run_train_quick.sh
```

**What this does:** Runs the quick training script that tests everything.

**If you get "permission denied" error:**
Type this first, then try again:
```bash
chmod +x run_train_quick.sh
```

### Step 4: Wait for It to Complete

The script will:
- Load the data
- Create features
- Train a model
- Show results

This takes 2-5 minutes. You'll see progress messages.

### Step 5: Check Results

When it's done, you'll see:
- Training metrics (MAE, RMSE)
- Feature importance
- A message saying "Quick training complete!"

## Visual Guide

```
Terminal Window:
┌─────────────────────────────────────────┐
│ Last login: ...                          │
│ doraweng@DoraWengs-MacBook-Pro ~ %      │
│                                          │
│ [Type commands here]                     │
│                                          │
└─────────────────────────────────────────┘
```

## Complete Example

Here's exactly what you'll type (copy-paste these lines one by one):

```bash
cd "/Users/doraweng/Documents/Kaggle Project/Hedege Fund-Time series forecast"
```

Press Enter, then:

```bash
./run_train_quick.sh
```

Press Enter and wait!

## Alternative: Use Python Directly

If the script doesn't work, you can run Python directly:

```bash
cd "/Users/doraweng/Documents/Kaggle Project/Hedege Fund-Time series forecast"
/usr/bin/python3 train_quick.py
```

## Common Issues

### "No such file or directory"
- Make sure you're in the right folder
- Check the folder name has no typos

### "Permission denied"
- Run: `chmod +x run_train_quick.sh`
- Then try again

### "Command not found"
- Make sure you typed `./run_train_quick.sh` (with the `./` at the start)

## What Happens Next?

After quick training works, you can:

1. **Full Training** (takes 30+ minutes):
   ```bash
   ./run_train.sh
   ```

2. **Generate Predictions**:
   ```bash
   ./run_predict.sh
   ```

## Need Help?

- Check `START_HERE.md` for quick overview
- Check `README.md` for detailed documentation
- All commands go in the Terminal window!
