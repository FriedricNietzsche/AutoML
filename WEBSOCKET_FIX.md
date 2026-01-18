# Quick Fix: WebSocket Connection Failure

## Problem
```
Firefox can't establish a connection to the server at ws://localhost:8000/ws/projects/sess_e5850b93fd28f8_1768722477808
[projectStore] incoming not array: undefined
```

## Root Cause
The backend was restarted, but the frontend is still trying to use old project IDs from before the restart. The backend has lost all in-memory project state.

## ✅ **IMMEDIATE FIX** (Do this now)

### Step 1: Clear Browser State
In your browser (Firefox), open DevTools Console and run:
```javascript
localStorage.clear();
sessionStorage.clear();
location.reload();
```

OR just **hard refresh**: Press **Cmd+Shift+R** (Mac) or **Ctrl+Shift+R** (Windows/Linux)

### Step 2: Start Fresh Workflow
1. Go to `http://localhost:5173/`
2. Enter a prompt: **"Build a classifier for the Titanic dataset"**
3. Click "Proceed"/"Start"
4. Watch the Console tab - you should see WebSocket connecting with a NEW project ID

### Step 3: Verify Connection
In Browser console, run:
```javascript
// Check WebSocket state
const store = window.__ZUSTAND_STORE__ || {};
console.log('Project ID:', store.projectId);
console.log('WS Status:', store.connectionStatus);
console.log('Dataset Sample:', store.datasetSample);
```

If `connectionStatus` is `"connected"`, you're good!

---

## 🔍 **Why This Happened**

1. **Backend restart** → All in-memory project states cleared
2. **Frontend kept old state** → localStorage still had old project ID
3. **Mismatch** → Frontend tries to connect to project that doesn't exist
4. **Connection fails** → No data flows, tables show zeros

---

## ✅ **Verify Fix Worked**

After clearing and reloading, check:
- [ ] WebSocket shows `readyState: 1` (OPEN) in console
- [ ] No more "can't establish connection" errors
- [ ] No more "[projectStore] incoming not array" errors  
- [ ] When you click "Proceed", backend terminal shows new WebSocket connection
- [ ] Data tableshould show REAL data, not zeros

---

## 🚀 **If Still Not Working**

### Check 1: Backend Terminal
You should see:
```
WebSocket connected for project sess_{NEW_ID}
Publishing event STAGE_STATUS to 1 subscribers
```

If you DON'T see this, the frontend isn't calling the backend properly.

### Check 2: Network Tab (WS)
Browser DevTools → Network → WS tab
- Should show connection to `ws://localhost:8000/ws/projects/sess_{NEW_ID}`
- Status should be "101 Switching Protocols" (success)
- Should see messages flowing

### Check 3: Frontend is Calling Backend
Browser DevTools → Network → Fetch/XHR tab
Look for POST request to:
- `/api/projects/{id}/confirm` - This triggers workflow stages

If you DON'T see these, the "Proceed" button isn't wired to the backend!

---

## 📝 **Next Debugging Steps** (If still broken after clear):

Tell me:
1. ✅ Did you clear localStorage and reload?
2. ✅ What's the NEW project ID in browser console?
3. ✅ Do you see WebSocket connection in backend terminal?
4. ✅ Does Network → WS tab show connected?
5. ✅ When you click "Proceed", what happens in backend terminal?
