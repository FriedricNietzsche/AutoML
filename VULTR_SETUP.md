# Vultr Cloud Training Setup

## What is this?

Train your models on Vultr GPU instances instead of your local machine!

**Benefits:**
- 🚀 **10-50x faster training** (NVIDIA A40 GPU vs CPU)
- 💰 **Pay-per-minute** (only charged when training)
- ☁️ **Automatic provisioning** (instances created and deleted automatically)
- 📊 **Same quality** (same models, same accuracy)

**Example:**
- Local CPU: 8 minutes for 3 epochs
- Vultr A40 GPU: 30 seconds for 3 epochs

## Setup Steps

### 1. Get Vultr API Key

1. Sign up at [vultr.com](https://www.vultr.com/) (you mentioned you have credits!)
2. Go to Account → API → Create API Key
3. Copy your API key (starts with a long string)

### 2. Configure Environment

Add to your `.env` file:

```bash
# Vultr Cloud Training
VULTR_API_KEY=your_vultr_api_key_here
USE_VULTR_TRAINING=true
```

### 3. Install Dependencies

```bash
cd backend
source .venv/bin/activate
pip install paramiko>=3.0.0
```

### 4. Test It!

Just train a model normally - the system will automatically:
1. Create a Vultr GPU instance (takes ~1-2 minutes)
2. Upload your training data
3. Train on GPU (30 seconds - 2 minutes)
4. Download the trained model
5. Delete the instance (stop paying)

## Pricing

**Vultr A40 GPU Instance:**
- $1.50/hour = $0.025/minute
- Typical training: 3-5 minutes = $0.08-0.13 per model
- Your free credits will last for 100+ models!

## Supported Tasks

Currently supported:
- ✅ Text classification (transformers)

Coming soon:
- ⏳ Image classification
- ⏳ Tabular models (XGBoost on GPU)

## Fallback

If Vultr training fails (API issues, no credits, etc.), the system automatically falls back to local CPU training.

## Disable Cloud Training

Set in `.env`:
```bash
USE_VULTR_TRAINING=false
```

## Monitoring

Watch the backend logs to see:
- Instance creation
- Training progress
- Download status
- Cleanup confirmation

## Troubleshooting

**Error: "VULTR_API_KEY environment variable not set"**
- Add your API key to `.env` file

**Error: "Instance failed to become ready"**
- Check your Vultr account has credits
- Try a different region (change `region="ewr"` to `region="lax"`)

**Training takes a long time**
- Instance provisioning: 1-2 minutes (one-time)
- Actual training: 30 seconds - 2 minutes
- Total: 2-4 minutes (still faster than 8 minutes local!)

## Cost Optimization

**Tips to minimize costs:**
1. Use smaller datasets for testing (automatic with MAX_SAMPLES=50000)
2. Reduce epochs (configured in code, default 3)
3. Instances auto-delete after training (no ongoing charges)
4. Only text classification uses cloud (tabular is fast locally)
