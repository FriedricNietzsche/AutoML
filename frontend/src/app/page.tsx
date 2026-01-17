import React from 'react';

const Page = () => {
    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
            <h1 className="text-4xl font-bold mb-4">Welcome to AutoML Agentic Builder</h1>
            <p className="text-lg mb-8">Log in to start building your machine learning models.</p>
            <button className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
                Log In
            </button>
        </div>
    );
};

export default Page;