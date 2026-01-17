import React from 'react';

const Layout = ({ children }) => {
    return (
        <div className="flex flex-col h-screen">
            <header className="bg-gray-800 text-white p-4">
                <h1 className="text-xl">AutoML Agentic Builder</h1>
            </header>
            <main className="flex-1 overflow-auto">
                {children}
            </main>
            <footer className="bg-gray-800 text-white p-4 text-center">
                <p>&copy; {new Date().getFullYear()} AutoML Agentic Builder</p>
            </footer>
        </div>
    );
};

export default Layout;