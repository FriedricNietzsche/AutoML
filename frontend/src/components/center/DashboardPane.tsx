import { useMemo } from 'react';
import { useProjectStore } from '../../store/projectStore';
import AIBuilderDashboard from './AIBuilderDashboard';
import DatasetImagePreview from './DatasetImagePreview';
import type { FileSystemNode } from '../../lib/types';
import { BarChart2, Image, TrendingUp } from 'lucide-react';

interface DashboardPaneProps {
   files: FileSystemNode[];
}

export default function DashboardPane({ files }: DashboardPaneProps) {
   const { currentStageId, datasetSample, trainingMetrics, profileSummary } = useProjectStore((state) => ({
      currentStageId: state.currentStageId,
      datasetSample: state.datasetSample,
      trainingMetrics: state.trainingMetrics,
      profileSummary: state.profileSummary,
   }));

   // Determine which view to show based on current stage
   const view = useMemo(() => {
      if (currentStageId === 'DATA_SOURCE' && datasetSample?.images && datasetSample.images.length > 0) {
         return 'dataset-images';
      }
      if (currentStageId === 'PROFILE_DATA' && profileSummary) {
         return 'profile';
      }
      if ((currentStageId === 'TRAIN' || currentStageId === 'REVIEW_EDIT') && trainingMetrics) {
         return 'training-metrics';
      }
      return 'default';
   }, [currentStageId, datasetSample, profileSummary, trainingMetrics]);

   return (
      <div className="h-full flex flex-col bg-replit-bg/30 backdrop-blur-xl">
         <div className="h-10 bg-replit-surface/60 backdrop-blur border-b border-replit-border/70 flex items-center px-4 justify-between shrink-0">
            <div className="flex items-center gap-2 text-replit-text font-medium text-sm">
               {view === 'dataset-images' && (
                  <>
                     <Image size={16} />
                     <span>Dataset Preview</span>
                  </>
               )}
               {view === 'profile' && (
                  <>
                     <BarChart2 size={16} />
                     <span>Data Profile</span>
                  </>
               )}
               {view === 'training-metrics' && (
                  <>
                     <TrendingUp size={16} />
                     <span>Training Metrics</span>
                  </>
               )}
               {view === 'default' && (
                  <>
                     <BarChart2 size={16} />
                     <span>Dashboard</span>
                  </>
               )}
            </div>
            <div className="text-xs text-replit-textMuted">
               {currentStageId.replace(/_/g, ' ')}
            </div>
         </div>
         <div className="flex-1 overflow-auto">
            {view === 'dataset-images' && (
               <DatasetImagePreview
                  images={datasetSample?.images}
                  columns={datasetSample?.columns}
                  nRows={datasetSample?.nRows}
               />
            )}
            {view === 'profile' && (
               <div className="p-6">
                  <h2 className="text-xl font-semibold mb-4">Dataset Profile</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                     <div className="bg-replit-surface/40 p-4 rounded-lg">
                        <div className="text-sm text-replit-textMuted">Rows</div>
                        <div className="text-2xl font-bold">{profileSummary?.n_rows.toLocaleString()}</div>
                     </div>
                     <div className="bg-replit-surface/40 p-4 rounded-lg">
                        <div className="text-sm text-replit-textMuted">Columns</div>
                        <div className="text-2xl font-bold">{profileSummary?.n_cols}</div>
                     </div>
                     <div className="bg-replit-surface/40 p-4 rounded-lg">
                        <div className="text-sm text-replit-textMuted">Missing %</div>
                        <div className="text-2xl font-bold">{profileSummary?.missing_pct.toFixed(1)}%</div>
                     </div>
                     <div className="bg-replit-surface/40 p-4 rounded-lg">
                        <div className="text-sm text-replit-textMuted">Warnings</div>
                        <div className="text-2xl font-bold">{profileSummary?.warnings.length || 0}</div>
                     </div>
                  </div>
                  {profileSummary?.warnings && profileSummary.warnings.length > 0 && (
                     <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
                        <h3 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Warnings</h3>
                        <ul className="list-disc list-inside space-y-1 text-sm">
                           {profileSummary.warnings.map((warning, idx) => (
                              <li key={idx}>{warning}</li>
                           ))}
                        </ul>
                     </div>
                  )}
               </div>
            )}
            {view === 'training-metrics' && trainingMetrics && (
               <div className="p-6">
                  <h2 className="text-xl font-semibold mb-4">Training Progress</h2>

                  {/* Progress Bar */}
                  {trainingMetrics.progress && (
                     <div className="mb-6">
                        <div className="flex justify-between text-sm mb-2">
                           <span>Step {trainingMetrics.progress.step} of {trainingMetrics.progress.steps}</span>
                           <span>{trainingMetrics.progress.phase}</span>
                        </div>
                        <div className="w-full bg-replit-surface/40 rounded-full h-2">
                           <div
                              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${(trainingMetrics.progress.step / trainingMetrics.progress.steps) * 100}%` }}
                           />
                        </div>
                     </div>
                  )}

                  {/* Metrics Cards */}
                  <div className="grid grid-cols-2 gap-4 mb-6">
                     {trainingMetrics.metricsHistory
                        .filter(m => m.split === 'test' || m.step === trainingMetrics.metricsHistory[trainingMetrics.metricsHistory.length - 1]?.step)
                        .reduce((acc, m) => {
                           if (!acc.find(x => x.name === m.name && x.split === m.split)) {
                              acc.push(m);
                           }
                           return acc;
                        }, [] as typeof trainingMetrics.metricsHistory)
                        .slice(0, 4)
                        .map((metric, idx) => (
                           <div key={idx} className="bg-replit-surface/40 p-4 rounded-lg">
                              <div className="text-sm text-replit-textMuted capitalize">{metric.name}</div>
                              <div className="text-2xl font-bold">{metric.value.toFixed(4)}</div>
                              <div className="text-xs text-replit-textMuted">{metric.split} set</div>
                           </div>
                        ))}
                  </div>

                  {/* Simple Loss Chart */}
                  {trainingMetrics.metricsHistory.filter(m => m.name === 'loss').length > 0 && (
                     <div className="bg-replit-surface/40 p-4 rounded-lg">
                        <h3 className="font-semibold mb-3">Loss Curve</h3>
                        <div className="h-48 flex items-end gap-1">
                           {trainingMetrics.metricsHistory
                              .filter(m => m.name === 'loss' && m.split === 'train')
                              .slice(-20)
                              .map((m, idx) => (
                                 <div
                                    key={idx}
                                    className="flex-1 bg-blue-500/60 rounded-t"
                                    style={{ height: `${Math.max(10, (1 - m.value) * 100)}%` }}
                                    title={`Step ${m.step}: ${m.value.toFixed(4)}`}
                                 />
                              ))}
                        </div>
                        <div className="text-xs text-replit-textMuted text-center mt-2">
                           Last 20 steps
                        </div>
                     </div>
                  )}

                  {/* Artifacts */}
                  {trainingMetrics.artifacts.length > 0 && (
                     <div className="mt-6">
                        <h3 className="font-semibold mb-3">Artifacts</h3>
                        <div className="space-y-2">
                           {trainingMetrics.artifacts.map((artifact) => (
                              <div key={artifact.id} className="bg-replit-surface/40 p-3 rounded-lg flex items-center justify-between">
                                 <div>
                                    <div className="font-medium">{artifact.name}</div>
                                    <div className="text-xs text-replit-textMuted">{artifact.type}</div>
                                 </div>
                                 {artifact.url && (
                                    <a
                                       href={artifact.url}
                                       target="_blank"
                                       rel="noopener noreferrer"
                                       className="text-xs bg-blue-500/20 hover:bg-blue-500/30 px-3 py-1 rounded text-blue-400 transition-colors"
                                    >
                                       View
                                    </a>
                                 )}
                              </div>
                           ))}
                        </div>
                     </div>
                  )}
               </div>
            )}
            {view === 'default' && <AIBuilderDashboard files={files} />}
         </div>
      </div>
   );
}
