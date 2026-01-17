import { Globe, Copy, ExternalLink } from 'lucide-react';

export default function PublishingPane() {
  return (
    <div className="h-full flex flex-col bg-replit-bg overflow-y-auto">
      <div className="max-w-2xl mx-auto p-8 w-full">
        <h2 className="text-2xl font-bold text-replit-text mb-6">
          Publish Your App
        </h2>

        {/* Deployment Status */}
        <div className="bg-replit-surface rounded-xl border border-replit-border p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center">
              <Globe className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <h3 className="text-replit-text font-medium">Ready to Deploy</h3>
              <p className="text-sm text-replit-textMuted">Your app is ready to be published</p>
            </div>
          </div>

          <div className="bg-replit-bg rounded-lg p-4 border border-replit-border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-replit-textMuted">Deployment URL</span>
              <button className="text-xs text-replit-accent hover:text-replit-accentHover flex items-center gap-1">
                <Copy className="w-3 h-3" />
                Copy
              </button>
            </div>
            <div className="flex items-center gap-2">
              <code className="text-sm text-replit-text font-mono">
                https://ai-builder-demo.replit.dev
              </code>
              <ExternalLink className="w-4 h-4 text-replit-textMuted" />
            </div>
          </div>
        </div>

        {/* Publish Options */}
        <div className="space-y-4 mb-6">
          <div className="bg-replit-surface rounded-xl border border-replit-border p-4">
            <label className="flex items-center gap-3 cursor-pointer">
              <input type="checkbox" className="w-4 h-4" defaultChecked />
              <div>
                <div className="text-sm text-replit-text font-medium">Auto-deploy on changes</div>
                <div className="text-xs text-replit-textMuted">Automatically redeploy when you push changes</div>
              </div>
            </label>
          </div>

          <div className="bg-replit-surface rounded-xl border border-replit-border p-4">
            <label className="flex items-center gap-3 cursor-pointer">
              <input type="checkbox" className="w-4 h-4" />
              <div>
                <div className="text-sm text-replit-text font-medium">Custom domain</div>
                <div className="text-xs text-replit-textMuted">Use your own domain name</div>
              </div>
            </label>
          </div>
        </div>

        {/* Publish Button */}
        <button className="w-full px-6 py-3 bg-replit-accent hover:bg-replit-accentHover text-white font-medium rounded-lg transition-colors">
          Publish to Production
        </button>

        <p className="text-xs text-replit-textMuted text-center mt-4">
          By publishing, you agree to make your app publicly accessible
        </p>
      </div>
    </div>
  );
}
