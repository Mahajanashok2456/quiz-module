# AI Quiz Generator

A full-stack AI-powered quiz generation application that allows users to enter any topic and receive AI-generated multiple-choice questions, take the quiz, and get instant scoring.

## Features

- **AI-Powered Quiz Generation**: Uses Google Gemini API to generate relevant multiple-choice questions
- **Dynamic Topic Support**: Enter any topic and get customized questions
- **Secure Storage**: Quiz data and results stored securely in MongoDB
- **Responsive Design**: Modern, mobile-friendly UI built with Next.js and TailwindCSS
- **Real-time Scoring**: Instant feedback and detailed results display

## Tech Stack

- **Frontend**: Next.js 15 (React) with TypeScript and TailwindCSS
- **Backend**: Python FastAPI with async support
- **Database**: MongoDB with Motor (async MongoDB driver)
- **AI Service**: Google Gemini API

## Project Structure

```
/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main application file
│   ├── models.py           # Database models and schemas
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js app directory
│   │   ├── components/    # React components
│   │   └── lib/           # Utilities and API client
│   └── package.json       # Node.js dependencies
└── README.md              # This file
```

## Setup Instructions

### Prerequisites

1. **Python 3.12+** (for backend)
2. **Node.js 18+** and **npm** (for frontend)
3. **MongoDB** (local installation or cloud instance)
4. **Google Gemini API Key**

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Configure environment variables
# Edit .env file with your settings:
# GEMINI_API_KEY=your_gemini_api_key_here
# MONGODB_URL=mongodb://localhost:27017
# DATABASE_NAME=ai_quiz_generator
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies (already done if using the provided setup)
npm install
```

### 3. Database Setup

Make sure MongoDB is running on your system:

```bash
# For local MongoDB installation
mongod

# Or use MongoDB Atlas (cloud) and update MONGODB_URL in .env
```

### 4. API Key Setup

1. Get your Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
2. Add it to `backend/.env`:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Running the Application

### Start Backend Server

```bash
cd backend
python main.py
```

The backend will start on `http://localhost:8000`

### Start Frontend Server

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Enter any topic in the text field (e.g., "Photosynthesis", "World War II", "JavaScript")
3. Click "Generate Quiz" to create AI-powered questions
4. Answer the multiple-choice questions
5. Submit your quiz to see your score and results
6. Take another quiz or return to the home page

## API Endpoints

- `POST /api/quiz/generate` - Generate a new quiz for a topic
- `POST /api/quiz/submit` - Submit answers and get scored results

## Development

### Adding New Features

- **Backend**: Add new endpoints in `main.py` and models in `models.py`
- **Frontend**: Create new components in `src/components/` and update the main page flow

### Database Schema

The application uses two main collections:

- `quizzes`: Stores complete quiz data with correct answers
- `results`: Stores user submissions and scores

## Troubleshooting

### Common Issues

1. **Backend won't start**: Check if MongoDB is running and API key is configured
2. **Quiz generation fails**: Verify Gemini API key is valid and has quota
3. **Frontend build errors**: Run `npm install` in the frontend directory

### Environment Variables

Make sure your `.env` file includes:

```
GEMINI_API_KEY=your_gemini_api_key_here
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=ai_quiz_generator
HOST=localhost
PORT=8000
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.
