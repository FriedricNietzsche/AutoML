'use client';

import * as React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import ThemeToggle from '@/components/ThemeToggle';
import { cn } from '@/lib/utils';

export default function Page() {
    const [modelName, setModelName] = React.useState('');
    const [task, setTask] = React.useState('');
    const [kaggleUrl, setKaggleUrl] = React.useState('');

    return (
        <div className="relative min-h-dvh overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0">
                {/* Dark-mode subtle glow */}
                <div className="hidden dark:block absolute -top-40 left-1/2 h-[720px] w-[720px] -translate-x-1/2 rounded-full bg-sky-500/10 blur-3xl" />
                <div className="hidden dark:block absolute -bottom-56 -left-56 h-[760px] w-[760px] rounded-full bg-indigo-500/10 blur-3xl" />
                <div className="hidden dark:block absolute -bottom-56 -right-72 h-[820px] w-[820px] rounded-full bg-fuchsia-500/8 blur-3xl" />
            </div>

            <div className="relative mx-auto w-full max-w-6xl px-6">
                {/* Header */}
                <div className="flex items-center justify-between py-6">
                    <div className="glass inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[12px] text-text/80 border border-border/60">
                        <span className="h-1.5 w-1.5 rounded-full bg-accent" />
                        <span>AutoML Builder</span>
                    </div>
                    <ThemeToggle />
                </div>

                {/* Hero + Panel */}
                <div className="mx-auto mt-6 max-w-4xl">
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ type: 'spring', stiffness: 180, damping: 18 }}
                    >
                        <div className="text-4xl md:text-5xl font-semibold tracking-tight text-text">
                            Build an AI model UI-first.
                        </div>
                        <div className="mt-3 text-sm md:text-base text-muted">
                            No backend required. Start a build session and watch the training simulator run.
                        </div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.08, type: 'spring', stiffness: 170, damping: 18 }}
                        className="mt-10"
                    >
                        <div className="glass mx-auto w-full max-w-xl rounded-2xl p-6 md:p-7 border border-border/60">
                            <div className="text-sm font-semibold text-text">Start a Build Session</div>
                            <div className="mt-1 text-xs text-muted">
                                Tell AutoAI what to build. We’ll simulate the pipeline and generate artifacts into the VFS.
                            </div>

                            <form
                                className="mt-6 space-y-4"
                                onSubmit={(e) => {
                                    e.preventDefault();
                                    // Task 1.2: placeholder behavior only.
                                }}
                            >
                                <div className="space-y-1.5">
                                    <label className="text-xs text-text/80">Model Name</label>
                                    <input
                                        value={modelName}
                                        onChange={(e) => setModelName(e.target.value)}
                                        placeholder="e.g. Review Sentiment Classifier"
                                        className={cn(
                                            'glass-input w-full rounded-xl px-3.5 py-2.5 text-sm text-text',
                                            'outline-none transition-colors',
                                            'placeholder:text-muted/70 focus:border-accent/60'
                                        )}
                                    />
                                </div>

                                <div className="space-y-1.5">
                                    <label className="text-xs text-text/80">What should the model do?</label>
                                    <textarea
                                        value={task}
                                        onChange={(e) => setTask(e.target.value)}
                                        placeholder="Describe the task and expected inputs/outputs..."
                                        rows={4}
                                        className={cn(
                                            'glass-input w-full resize-none rounded-xl px-3.5 py-2.5 text-sm text-text',
                                            'outline-none transition-colors',
                                            'placeholder:text-muted/70 focus:border-accent/60'
                                        )}
                                    />
                                    <div className="text-[11px] text-muted">
                                        Required. This drives the simulated API schema and demo inference.
                                    </div>
                                </div>

                                <div className="space-y-1.5">
                                    <label className="text-xs text-text/80">Kaggle dataset link (optional)</label>
                                    <input
                                        value={kaggleUrl}
                                        onChange={(e) => setKaggleUrl(e.target.value)}
                                        placeholder="https://kaggle.com/datasets/..."
                                        className={cn(
                                            'glass-input w-full rounded-xl px-3.5 py-2.5 text-sm text-text',
                                            'outline-none transition-colors',
                                            'placeholder:text-muted/70 focus:border-accent/60'
                                        )}
                                    />
                                    <div className="text-[11px] text-muted">
                                        Paste a Kaggle dataset link. Leave blank and AutoAI will choose one.
                                    </div>
                                </div>

                                <div className="pt-2 flex items-center justify-between">
                                    <div className="text-[11px] text-muted">Build starts automatically in the workspace.</div>
                                    <motion.button
                                        type="submit"
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        className={cn(
                                            'inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium',
                                            'bg-control text-text border border-border/60',
                                            'hover:bg-panel transition-colors'
                                        )}
                                    >
                                        <span>Start Build</span>
                                        <ArrowRight className="h-4 w-4" />
                                    </motion.button>
                                </div>
                            </form>
                        </div>
                    </motion.div>
                </div>
            </div>
        </div>
    );
}