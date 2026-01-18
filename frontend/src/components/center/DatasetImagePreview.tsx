import React from 'react';

interface DatasetImagePreviewProps {
    images?: string[];
    columns?: string[];
    nRows?: number;
}

export default function DatasetImagePreview({ images, columns, nRows }: DatasetImagePreviewProps) {
    if (!images || images.length === 0) {
        return (
            <div className="p-8 text-center text-gray-500">
                <p>No dataset images available yet...</p>
                {columns && columns.length > 0 && (
                    <div className="mt-4 text-sm">
                        <p>{nRows} rows, {columns.length} columns</p>
                        <p className="text-xs text-gray-400 mt-2">{columns.slice(0, 5).join(', ')}</p>
                    </div>
                )}
            </div>
        );
    }

    // Limit to 3 images for preview
    const displayImages = images.slice(0, 3);

    return (
        <div className="p-6">
            <h2 className="text-xl font-semibold mb-4">Dataset Sample</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                {displayImages.map((imgUrl, idx) => (
                    <div key={idx} className="border rounded-lg overflow-hidden shadow-sm">
                        <img
                            src={imgUrl}
                            alt={`Sample ${idx + 1}`}
                            className="w-full h-48 object-cover"
                            onError={(e) => {
                                (e.target as HTMLImageElement).src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23ddd" width="200" height="200"/%3E%3Ctext fill="%23999" x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle"%3ENo Image%3C/text%3E%3C/svg%3E';
                            }}
                        />
                        <div className="p-2 bg-gray-50 text-xs text-gray-600">
                            Sample {idx + 1}
                        </div>
                    </div>
                ))}
            </div>
            {nRows && (
                <div className="text-sm text-gray-600">
                    <p>Total: {nRows} images</p>
                    {images.length > 3 && <p className="text-xs text-gray-400">Showing first 3 sample images</p>}
                </div>
            )}
        </div>
    );
}
