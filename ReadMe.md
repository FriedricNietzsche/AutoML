# AutoML Agentic Builder

## Overview
The AutoML Agentic Builder is a chat-first web application designed to assist users in building machine learning models through an interactive and intuitive interface. The application guides users through a series of stages, allowing them to confirm actions and visualize the progress of their machine learning projects in real-time.

## Project Structure
The project is divided into two main directories: `frontend` and `backend`.

### Frontend
The frontend is built using Next.js with TypeScript, Tailwind CSS, and Zustand for state management. It provides a rich user interface that allows users to interact with the application seamlessly.

- **src/app**: Contains the main application pages.
- **src/components**: Contains reusable UI components.
- **src/lib**: Contains utility functions for API calls, WebSocket management, and state management.
- **public**: Contains static assets.

### Backend
The backend is built using FastAPI and manages the application's business logic, API endpoints, and WebSocket connections. It handles user authentication, project management, and the orchestration of the machine learning pipeline.

- **app**: Contains the main application logic, including API routes, WebSocket handling, and orchestrator logic.
- **data**: Contains runtime data storage for projects.
- **requirements.txt**: Lists the Python dependencies required for the backend.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher
- PostgreSQL (for production use, optional)

### Backend Setup
1. Navigate to the `backend` directory:
   ```
   cd backend
   ```
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the FastAPI application:
   ```
   uvicorn app.main:app --reload
   ```

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```
   cd frontend
   ```
2. Install the required Node.js packages:
   ```
   npm install
   ```
3. Run the Next.js application:
   ```
   npm run dev
   ```

## Usage
- Users can log in using Supabase authentication.
- After logging in, users can enter prompts to initiate the model-building process.
- The application will guide users through the stages of data collection, preprocessing, training, review, and export.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.