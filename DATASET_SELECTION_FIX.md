# Dataset Selection Flow - Auto-Download Fix

## Problem
Datasets were downloading immediately when clicked instead of waiting for the "Next" button.

## Root Cause
The `/dataset/select` endpoint contained full download logic that executed immediately when the user clicked a dataset card.

## Solution
Separated dataset selection into two distinct steps:

### 1. **Selection** (`/dataset/select`)
- **When**: User clicks on a dataset card
- **What it does**: Stores the selected dataset in project context
- **Backend**: Returns immediately with `{status: "ok", message: "Dataset selected - click Next to download"}`
- **No download happens**

### 2. **Download** (`/dataset/download`) 
- **When**: User clicks "Confirm & Continue" (Next) button
- **What it does**: Downloads the dataset from HuggingFace, saves as CSV
- **Backend**: Publishes progress events via WebSocket
- **Returns**: `{status: "ok", path: "/path/to/dataset.csv"}`

## Changes Made

### Backend (`backend/app/api/data.py`)

1. **Modified `/dataset/select` endpoint** (lines 204-273):
   - Now only stores selection in project context
   - Checks if upload is required (`upload_csv` option)
   - Returns immediately without downloading

2. **Created new `/dataset/download` endpoint** (lines 275-435):
   - Validates a dataset was selected
   - Downloads from HuggingFace using selected dataset ID
   - Samples large datasets to 500 rows max
   - Saves as CSV in project directory
   - Publishes progress events

### Frontend (`frontend/src/hooks/useBackendPipeline.ts`)

1. **Modified `confirmStage()` function** (lines 160-212):
   - Detects when confirming DATA_SOURCE stage
   - Calls `/dataset/download` before confirming
   - Shows download progress via WebSocket events
   - Handles errors gracefully

## User Flow (After Fix)

1. ✅ User clicks dataset card → UI shows selected (NO download)
2. ✅ User reviews selection, can change their mind
3. ✅ User clicks "Confirm & Continue" → Download starts
4. ✅ Progress shown via WebSocket events
5. ✅ Dataset ready for training

## Special Cases

### CSV Upload Option
- Dataset ID: `upload_csv`
- Flag: `is_upload_prompt: true`
- Returns: `{requires_upload: true}`
- Does NOT trigger download (separate upload flow needed)

### Already Downloaded
- `/dataset/download` checks if file exists
- Returns immediately with existing path if found
- No re-download needed

## Testing Checklist

- [ ] Click dataset card → Shows selected, NO download
- [ ] Click different dataset → Updates selection
- [ ] Click "Confirm & Continue" → Download starts
- [ ] Progress events appear during download
- [ ] Download completes successfully
- [ ] CSV file saved in project directory
- [ ] Upload CSV option returns requires_upload flag

## Future Improvements

1. **CSV Upload UI**: Create separate file upload component
2. **Download Caching**: Cache downloads across projects
3. **Cancel Download**: Add ability to cancel in-progress download
4. **Preview Data**: Show dataset preview before downloading
