import React from 'react';

interface DatasetPreviewProps {
    projectId: string;
    dataset?: {
        name: string;
        description: string;
        sampleData: Array<{ [key: string]: any }>;
    };
}

const DatasetPreview: React.FC<DatasetPreviewProps> = ({ projectId, dataset }) => {
    if (!dataset) {
        return (
            <div className="p-4 border rounded shadow">
                <p className="text-gray-600">No dataset available for project {projectId}</p>
            </div>
        );
    }

    return (
        <div className="p-4 border rounded shadow">
            <h2 className="text-xl font-bold">{dataset.name}</h2>
            <p className="text-gray-600">{dataset.description}</p>
            <h3 className="mt-4 text-lg font-semibold">Sample Data</h3>
            <table className="min-w-full mt-2 border">
                <thead>
                    <tr>
                        {Object.keys(dataset.sampleData[0]).map((key) => (
                            <th key={key} className="border px-4 py-2">{key}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {dataset.sampleData.map((row, index) => (
                        <tr key={index}>
                            {Object.values(row).map((value, idx) => (
                                <td key={idx} className="border px-4 py-2">{value}</td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default DatasetPreview;