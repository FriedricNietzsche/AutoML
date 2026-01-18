# ROOT CAUSE FOUND: WebSocket Reconnection Loop

## 🎯 The Real Problem

The WebSocket connection is **constantly reconnecting** because of a React useEffect dependency issue!

### What's Happening:

Looking at `/frontend/src/components/shell/AppShell.tsx` lines 158-161:

```typescript
useEffect(() => {
  connectProject({ projectId, wsBase });
  hydrate();
}, [connectProject, hydrate, projectId, wsBase]);  // ← TOO MANY DEPENDENCIES!
```

### The Sequence:

1. **Page loads** → useEffect runs → Connects WebSocket with project ID `sess_A` ✅
2. **Something updates** `projectId`, `connectProject`, or `hydrate` reference
3. **useEffect runs AGAIN** → Closes old WebSocket, opens new one
4. **Race condition**: Old WebSocket cleanup happens while new connection is starting
5. **Error**: `"WebSocket is not connected. Need to call 'accept' first."`

### Evidence from Backend Logs:

```
02:58:42,178 - WebSocket connected for project sess_ce5cd8e260a578_1768723122052
02:58:42,202 - Publishing event STAGE_STATUS
02:58:42,205 - Subscriber removed  ← OLD CONNECTION CLEANUP
02:58:42,205 - WebSocket disconnected
02:58:42,208 - ERROR: WebSocket is not connected
02:58:42,343 - WebSocket connected ← NEW CONNECTION (140ms later!)
```

The connection is **created → destroyed → recreated** in under 200ms!

## ✅ The Fix

Two options:

### Option 1: Stabilize Dependencies (Quick Fix)

Wrap `connectProject` and `hydrate` in `useCallback` to prevent them from changing on every render.

### Option 2: Only Reconnect When Project ID Changes (Better)

Change the useEffect to only depend on `projectId` and `wsBase`:

```typescript
useEffect(() => {
  connectProject({ projectId, wsBase });
  hydrate();
}, [project Id, wsBase]);  // Remove connectProject and hydrate from dependencies
```

This way, it only reconnects when the **actual project ID changes**, not when function references change.

## 🔧 Applying the Fix

I'll update `AppShell.tsx` to remove the unnecessary dependencies.

## 📊 Expected Result After Fix

Backend logs should show:
```
WebSocket connected for project sess_{id}
connection open
Publishing event STAGE_STATUS to 1 subscribers
[stays connected - no immediate disconnect]
```

Frontend should show:
- ✅ WebSocket stays connected
- ✅ No reconnection loop
- ✅ Data flows properly
