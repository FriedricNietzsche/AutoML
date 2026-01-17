import React from 'react';

interface ConfirmBarProps {
    isWaitingForConfirmation?: boolean;
    onConfirm?: () => void;
}

const ConfirmBar: React.FC<ConfirmBarProps> = ({ isWaitingForConfirmation = false, onConfirm }) => {
    return (
        <div className="confirm-bar">
            {isWaitingForConfirmation && (
                <button 
                    className="confirm-button" 
                    onClick={() => onConfirm?.()}
                >
                    Confirm
                </button>
            )}
        </div>
    );
};

export default ConfirmBar;