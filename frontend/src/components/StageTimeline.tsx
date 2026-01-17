import React from 'react';

const stages = [
    { id: 'PARSE_INTENT', label: 'Parse Intent', status: 'completed' },
    { id: 'DATA_SOURCE', label: 'Data Source', status: 'in-progress' },
    { id: 'PROFILE_DATA', label: 'Profile Data', status: 'pending' },
    { id: 'PREPROCESS', label: 'Preprocess', status: 'pending' },
    { id: 'MODEL_SELECT', label: 'Model Select', status: 'pending' },
    { id: 'TRAIN', label: 'Train', status: 'pending' },
    { id: 'REVIEW_EDIT', label: 'Review/Edit', status: 'pending' },
    { id: 'EXPORT', label: 'Export', status: 'pending' },
];

const StageTimeline: React.FC<{ projectId?: string }> = () => {
    return (
        <div className="flex flex-col">
            <h2 className="text-lg font-bold mb-2">Stage Timeline</h2>
            <ul className="space-y-2">
                {stages.map(stage => (
                    <li key={stage.id} className={`p-2 rounded ${stage.status === 'completed' ? 'bg-green-200' : stage.status === 'in-progress' ? 'bg-yellow-200' : 'bg-gray-200'}`}>
                        <span className="font-semibold">{stage.label}</span> - <span className="text-sm">{stage.status}</span>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default StageTimeline;