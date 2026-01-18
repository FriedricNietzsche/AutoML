# AutoML Quick Start Guide

## 🚀 Getting Started in 3 Minutes

### Prerequisites
- Python 3.8+ installed
- Node.js 14+ installed
- Virtual environment activated (backend)

### Step 1: Start the Backend (Terminal 1)

```bash
# Navigate to backend directory
cd /Users/krisviraujla/AutoML/backend

# Activate virtual environment
source venv/bin/activate  # or: source .venv/bin/activate

# Start the server
python3 -m uvicorn app.main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 2: Start the Frontend (Terminal 2)

```bash
# Navigate to frontend directory
cd /Users/krisviraujla/AutoML/frontend

# Install dependencies (if not done)
npm install

# Start dev server
npm run dev
```

**Expected Output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Step 3: Open the Application

1. Open your browser
2. Navigate to: **http://localhost:5173**
3. You should see the AutoML interface

### Step 4: Run the Demo Workflow

#### Option A: Via Browser Console

1. Open browser DevTools (F12 or Cmd+Option+I)
2. Go to Console tab
3. Paste and run:

```javascript
fetch('http://localhost:8000/api/demo/run/demo?prompt=Build%20me%20a%20classifier%20for%20cat/dog', {
  method: 'POST'
}).then(r => r.json()).then(console.log);
```

#### Option B: Via Terminal (curl)

```bash
curl -X POST "http://localhost:8000/api/demo/run/demo?prompt=Build%20me%20a%20classifier%20for%20cat/dog"
```

### Step 5: Watch the Magic Happen 🎩✨

1. **Click on the "Dashboard" tab** in the AutoML interface
2. **Open WebSocket inspector** (DevTools → Network → WS filter)
3. **Watch the stages progress:**

   **Stage 1 - Data Source** (10-15s)
   - Dataset images appear in Dashboard
   - 3 cat/dog sample images displayed

   **Stage 2 - Profile Data** (2-3s)
   - Profile metrics cards appear
   - Shows rows, columns, missing %

   **Stage 3 - Model Select** (1s)
   - Model candidates selected (logged to WS)

   **Stage 4 - Training** (10-15s) 🔥
   - Progress bar animates
   - Loss curve updates in real-time
   - Metrics cards show live values
   - Artifacts appear as generated

   **Stage 5 - Export** (2-3s)
   - Notebook generated
   - Export bundle ready

**Total time: ~30-40 seconds**

## 🔍 What to Look For

### In the Dashboard Tab

✅ **Dataset Images** (Stage 1)
- Grid of 3 images
- Row count displayed
- Column count displayed

✅ **Profile Metrics** (Stage 2)
- Cards showing: Rows, Columns, Missing %, Warnings
- Yellow warning box if issues detected

✅ **Training Progress** (Stage 4)
- Blue progress bar moving
- Metric cards updating (Loss, Accuracy, etc.)
- Loss curve growing from left to right
- Artifacts appearing in list

### In the WebSocket Tab (DevTools)

✅ **Event Stream**
```
→ HELLO
→ STAGE_STATUS (PARSE_INTENT)
→ PROMPT_PARSED
→ STAGE_STATUS (DATA_SOURCE)
→ DATASET_SAMPLE_READY (with images array)
→ STAGE_STATUS (PROFILE_DATA)
→ PROFILE_SUMMARY
→ STAGE_STATUS (MODEL_SELECT)
→ MODEL_CANDIDATES
→ STAGE_STATUS (TRAIN)
→ TRAIN_RUN_STARTED
→ LOSS_SURFACE_SPEC_READY
→ GD_PATH_STARTED
→ TRAIN_PROGRESS (× 50 times)
→ GD_PATH_UPDATE (× 10 times)
→ METRIC_SCALAR (× many times)
→ TRAIN_RUN_FINISHED
→ STAGE_STATUS (REVIEW_EDIT)
→ NOTEBOOK_READY
→ STAGE_STATUS (EXPORT)
→ EXPORT_READY
```

### In the Console Tab (DevTools)

✅ **Store State** (optional check)
```javascript
// Inspect the store
window.projectStore = window.useProjectStore || {}
console.log('Dataset:', window.projectStore.getState?.().datasetSample)
console.log('Training:', window.projectStore.getState?.().trainingMetrics)
console.log('GD Path:', window.projectStore.getState?.().gdPath)
```

## 🐛 Troubleshooting

### Backend Won't Start

**Error:** `python: command not found`
**Fix:** Use `python3` instead:
```bash
python3 -m uvicorn app.main:app --reload --port 8000
```

**Error:** `No module named 'app'`
**Fix:** Ensure you're in the backend directory and venv is activated:
```bash
cd /Users/krisviraujla/AutoML/backend
source venv/bin/activate
```

**Error:** `ModuleNotFoundError: No module named 'fastapi'`
**Fix:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Won't Start

**Error:** `npm: command not found`
**Fix:** Install Node.js from https://nodejs.org/

**Error:** Dependencies not installed
**Fix:**
```bash
cd /Users/krisviraujla/AutoML/frontend
npm install
```

**Error:** Port 5173 already in use
**Fix:** Kill the existing process or use a different port:
```bash
npm run dev -- --port 5174
```

### WebSocket Won't Connect

**Error:** Connection refused
**Fix:** Ensure backend is running on port 8000:
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","service":"automl-backend"}
```

**Error:** CORS errors
**Fix:** Backend has CORS enabled for all origins, but verify you're accessing frontend from http://localhost:5173

### Dashboard Shows Nothing

**Symptom:** Dashboard tab is empty or shows default view
**Fix:**
1. Check WebSocket is connected (green indicator in TopBar)
2. Verify demo workflow was triggered (check backend logs)
3. Check browser console for errors
4. Verify you clicked the "Dashboard" tab

### Demo Workflow Fails

**Error:** HuggingFace dataset download fails
**Symptom:** Error in backend logs about dataset not found
**Fix:** Demo uses public datasets, but network issues can occur. Try again or use a different dataset.

**Error:** Training fails
**Symptom:** TRAIN_RUN_FINISHED with status "failed"
**Fix:** Check backend logs for error details. Common issues:
- Insufficient memory (reduce TRAIN_MAX_ROWS in env)
- Missing dependencies (ensure sklearn, pandas installed)

## 🎯 Quick Verification Checklist

Run through this checklist to verify everything is working:

- [ ] Backend starts without errors
- [ ] Frontend starts and shows UI
- [ ] Can access http://localhost:5173
- [ ] WebSocket connects (check TopBar indicator)
- [ ] Demo endpoint responds (curl test works)
- [ ] Dashboard tab exists and is clickable
- [ ] Dataset images appear after ~15 seconds
- [ ] Profile metrics appear after ~20 seconds
- [ ] Training progress bar animates
- [ ] Loss curve updates during training
- [ ] Artifacts appear in dashboard
- [ ] No errors in browser console
- [ ] No errors in backend logs

## 🎉 Success!

If you see all the above, congratulations! The AutoML WebSocket integration is working perfectly.

Your cat/dog classifier was built end-to-end with real-time updates streaming to the frontend!

## 📚 Next Steps

1. **Explore the Code:**
   - `backend/app/api/demo.py` - Full workflow orchestration
   - `frontend/src/store/projectStore.ts` - Event handling
   - `frontend/src/components/center/DashboardPane.tsx` - UI updates

2. **Try Different Prompts:**
   ```bash
   # Regression task
   curl -X POST "http://localhost:8000/api/demo/run/demo2?prompt=Predict house prices from features"
   
   # Different classification
   curl -X POST "http://localhost:8000/api/demo/run/demo3?prompt=Build spam classifier"
   ```

3. **Add More Visualization:**
   - Wire up GradientDescentViz component (if it exists)
   - Add matrix→vector transformation animation
   - Create notebook viewer
   - Add export button with download

4. **Customize the Workflow:**
   - Modify `demo.py` to change dataset source
   - Adjust training parameters
   - Add new event types
   - Create custom visualization components

## 💡 Pro Tips

- **Use DevTools extensively** - The WebSocket and Console tabs are your best friends
- **Watch backend logs** - They show exactly what's happening server-side
- **Check the store state** - Store has ALL the data, components just render it
- **Stage transitions are key** - Each stage emits specific events, dashboard adapts

---

**Need help?** Check `INTEGRATION_SUMMARY.md` for detailed technical documentation.

**Found a bug?** Check browser console and backend logs for error details.
