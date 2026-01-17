import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export type LossPoint = {
  epoch: number;
  train_loss: number;
  val_loss: number;
};

interface TrainingLossVisualizerProps {
  data: LossPoint[];
}

export default function TrainingLossVisualizer({ data }: TrainingLossVisualizerProps) {
  const currentTrain = data.length > 0 ? data[data.length - 1]?.train_loss : null;
  const currentVal = data.length > 0 ? data[data.length - 1]?.val_loss : null;
  const currentEpoch = data.length > 0 ? data[data.length - 1]?.epoch : 0;

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-replit-border/60 bg-replit-surface/35 backdrop-blur p-6 pb-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold text-replit-text">Training Loss</div>
          </div>
          <div className="flex items-center gap-2 text-xs text-replit-textMuted">
            <span>Epoch</span>
            <span className="font-mono text-replit-text">{currentEpoch}</span>
          </div>
        </div>

        <div className="bg-replit-bg/30 rounded-xl p-4 mb-4 relative overflow-hidden border border-replit-border/60">
          {data.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center bg-replit-bg/40 backdrop-blur-sm z-10">
              <div className="text-center">
                <div className="w-10 h-10 border-4 border-replit-accent border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                <p className="text-replit-textMuted">Waiting for data…</p>
              </div>
            </div>
          )}
          <ResponsiveContainer width="100%" height={560}>
            <LineChart data={data} margin={{ top: 5, right: 24, left: 6, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.25)" />
              <XAxis dataKey="epoch" stroke="rgb(var(--replit-text-muted))" />
              <YAxis stroke="rgb(var(--replit-text-muted))" domain={[0, 'auto']} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(20,24,36,0.9)',
                  border: '1px solid rgba(148,163,184,0.25)',
                  borderRadius: '8px',
                  color: '#fff',
                }}
              />
              <Legend wrapperStyle={{ paddingTop: '12px' }} iconType="line" />
              <Line
                type="monotone"
                dataKey="train_loss"
                stroke="rgb(var(--replit-accent))"
                strokeWidth={3}
                name="Training Loss"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke="rgb(var(--replit-warning))"
                strokeWidth={3}
                name="Validation Loss"
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-replit-accent/10 border border-replit-accent/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-4 h-1 bg-replit-accent rounded"></div>
              <span className="text-replit-accent font-semibold">Training Loss</span>
            </div>
            <p className="text-replit-text text-2xl font-mono">{currentTrain !== null ? currentTrain.toFixed(3) : '—'}</p>
          </div>
          <div className="bg-replit-warning/10 border border-replit-warning/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-4 h-1 bg-replit-warning rounded"></div>
              <span className="text-replit-warning font-semibold">Validation Loss</span>
            </div>
            <p className="text-replit-text text-2xl font-mono">{currentVal !== null ? currentVal.toFixed(3) : '—'}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
