'use client';

import React, { useState } from 'react';
import ChatPanel from '@/components/ChatPanel';
import StageTimeline from '@/components/StageTimeline';
import ConfirmBar from '@/components/ConfirmBar';
import DatasetPreview from '@/components/DatasetPreview';
import ProfilingPanel from '@/components/ProfilingPanel';
import TrainingDashboard from '@/components/TrainingDashboard';
import FileExplorer from '@/components/FileExplorer';
import ArtifactsPanel from '@/components/ArtifactsPanel';
import ConsoleLog from '@/components/ConsoleLog';

interface ProjectPageProps {
    params: {
        projectId: string;
    };
}

const ProjectPage = ({ params }: ProjectPageProps) => {
    const { projectId } = params;
    const [isWaitingForConfirmation, setIsWaitingForConfirmation] = useState(false);

    const handleConfirm = () => {
        // Handle confirmation logic here
        console.log('Confirmed');
        setIsWaitingForConfirmation(false);
    };

    return (
        <div className="flex h-screen">
            <div className="w-1/4 p-4 border-r">
                <ChatPanel />
                <StageTimeline projectId={projectId} />
                <ConfirmBar 
                    isWaitingForConfirmation={isWaitingForConfirmation}
                    onConfirm={handleConfirm}
                />
            </div>
            <div className="w-1/2 p-4 border-r">
                <DatasetPreview projectId={projectId} />
                <ProfilingPanel projectId={projectId} />
                <TrainingDashboard projectId={projectId} />
            </div>
            <div className="w-1/4 p-4">
                <FileExplorer projectId={projectId} />
                <ArtifactsPanel projectId={projectId} />
                <ConsoleLog projectId={projectId} />
            </div>
        </div>
    );
};

export default ProjectPage;