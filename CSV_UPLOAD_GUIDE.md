# How to Upload Your Own CSV Dataset

## Quick Answer

**To upload your own CSV file:**

1. When the system shows "Found 0 datasets", look for the **"📤 Upload your own dataset"** option
2. Click on it
3. A file upload dialog will appear
4. Click "Click to browse" or drag your CSV file into the upload area
5. Select your `.csv` file
6. Wait for upload to complete
7. Click "Confirm & Continue" to proceed

## Step-by-Step Guide

### Option 1: Using the Upload Dataset Button

When you're at the dataset selection stage and see:
```
Found 0 datasets
• Upload your own dataset
```

1. **Click** on "Upload your own dataset"
2. **Upload Dialog appears** with a drag-and-drop area
3. **Select your CSV file** by:
   - Clicking "Click to browse" and choosing a file
   - OR dragging and dropping your `.csv` file
4. **Wait** for the upload progress (you'll see a spinner)
5. **Success** - The file is uploaded to the backend
6. **Click** "Confirm & Continue" to move to the next stage

### Option 2: Direct API Upload (Advanced)

If you prefer to upload via API:

```bash
curl -X POST http://localhost:8000/api/projects/demo-project/upload \
  -F "file=@/path/to/your/dataset.csv"
```

Response:
```json
{
  "status": "ok",
  "rows": 1000,
  "columns": ["feature1", "feature2", "target"]
}
```

## CSV File Requirements

### Format
- ✅ **File extension**: `.csv`
- ✅ **Encoding**: UTF-8 recommended
- ✅ **Headers**: First row should contain column names
- ✅ **Size**: Reasonable size (will be sampled to 500 rows for demo)

### Example CSV Structure

**Classification** (e.g., Titanic survival):
```csv
age,sex,fare,survived
22,male,7.25,0
38,female,71.28,1
26,female,7.92,1
```

**Regression** (e.g., House prices):
```csv
bedrooms,bathrooms,sqft,price
3,2,1500,250000
4,3,2000,350000
2,1,800,150000
```

## What Happens After Upload

1. **Backend Processing**:
   - File is saved to `/backend/assets/projects/demo-project/`
   - Reads first 500 rows (for performance)
   - Generates dataset sample and profile
   - Publishes events via WebSocket

2. **Frontend Updates**:
   - Shows "✅ Dataset uploaded"
   - Enables "Confirm & Continue" button
   - Displays dataset stats (rows, columns)

3. **Next Steps**:
   - Click "Confirm & Continue"
   - System auto-detects task type (classification/regression)
   - Proceeds to model selection stage

## Current Implementation Status

### ✅ Working
- Backend `/upload` endpoint
- File validation and storage
- CSV parsing and sampling
- Context storage in orchestrator
- Event publishing

### ✅ NEW - Just Added
- Upload dialog UI component
- File picker with drag-and-drop
- Upload progress indicator
- Automatic selection when upload_csv clicked
- Cancel functionality

### 📊 Backend Endpoint

**URL**: `POST /api/projects/{project_id}/upload`

**Headers**: `multipart/form-data`

**Body**: Form data with `file` field

**Response**:
```json
{
  "status": "ok",
  "rows": 500,
  "columns": ["col1", "col2", "target"]
}
```

## Troubleshooting

### "Upload button doesn't appear"
**Cause**: No datasets found by AI search

**Solution**: This is expected - when AI finds 0 relevant datasets, the upload option automatically appears

### "File upload fails"
**Possible causes**:
- File is not `.csv` format
- File is corrupted or empty
- File is too large (>100MB)
- Missing headers in first row

**Solution**: Check your CSV file format and try again

### "Upload dialog doesn't show"
**Cause**: Frontend not detecting `requires_upload` flag

**Check**: Backend response should include:
```json
{
  "status": "ok",
  "requires_upload": true
}
```

## File Location After Upload

Your uploaded file is stored at:
```
/backend/assets/projects/demo-project/your_file.csv
```

Sample preview is saved at:
```
/backend/assets/projects/demo-project/sample.csv
```

## Next: Using Your Dataset

After successful upload:

1. **Confirm Stage** - Click "Confirm & Continue"
2. **Profile Data** - System analyzes your columns
3. **Model Select** - AI recommends models based on your data
4. **Train** - Model trains on your dataset
5. **Export** - Download trained model

## Example: Complete Upload Flow

```
User: "Build ML model to predict house prices"
  ↓
AI Search: "No datasets found for 'house prices'"
  ↓
System: Shows "📤 Upload your own dataset"
  ↓
User: Clicks upload option
  ↓
Dialog: File picker appears
  ↓
User: Selects "housing_data.csv"
  ↓
Upload: File sent to backend (shows progress)
  ↓
Backend: Saves file, reads 500 rows, analyzes columns
  ↓
Success: "✅ Dataset uploaded: 500 rows, 10 columns"
  ↓
User: Clicks "Confirm & Continue"
  ↓
Next Stage: Model selection begins
```

## Code Reference

### Frontend Handler
Location: `/frontend/src/components/center/loader/RealBackendLoader.tsx`

```typescript
const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
  const file = event.target.files?.[0];
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`/api/projects/${projectId}/upload`, {
    method: 'POST',
    body: formData,
  });
}
```

### Backend Handler
Location: `/backend/app/api/data.py`

```python
@router.post("/{project_id}/upload")
async def upload_dataset(project_id: str, file: UploadFile = File(...)):
    # Save file
    dest = project_dir / file.filename
    
    # Parse CSV
    df = pd.read_csv(dest, nrows=500)
    
    # Store in context
    context["selected_dataset"] = {
        "source": "upload",
        "filename": file.filename,
        "path": str(dest),
        "rows": len(df),
        "columns": list(df.columns)
    }
```

## Summary

✅ **Upload Now Works!**
- Click "Upload your own dataset" when shown
- Upload dialog appears automatically
- Select your CSV file
- Upload completes and stores in backend
- Proceed with "Confirm & Continue"

🎉 Your dataset is ready for ML training!
