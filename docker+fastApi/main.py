from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Modelo de datos para una película
class Movie(BaseModel):
    title: str
    year: int
    genre: str

# Base de datos ficticia
movies_db = [
    {"title": "The Shawshank Redemption", "year": 1994, "genre": "Drama"},
    {"title": "The Godfather", "year": 1972, "genre": "Crime"},
    {"title": "The Dark Knight", "year": 2008, "genre": "Action"}
]

@app.get("/movies")
async def get_movies():
    """Devuelve la lista de películas."""
    return {"movies": movies_db}

@app.post("/movies")
async def add_movie(movie: Movie):
    """Agrega una nueva película a la base de datos."""
    movies_db.append(movie.dict())
    return {"message": "Película agregada con éxito", "movie": movie}

@app.delete("/movies/{title}")
async def delete_movie(title: str):
    """Elimina una película de la base de datos por título."""
    for movie in movies_db:
        if movie["title"] == title:
            movies_db.remove(movie)
            return {"message": f"Película '{title}' eliminada con éxito"}
    raise HTTPException(status_code=404, detail="Película no encontrada")
