from fastapi import FastAPI
import uvicorn
from  pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

app = FastAPI(debug=True)

@app.get('/{name}')
def greet(name:str):
    return {"Welcome {}! Enter the values to predict breast cancer.".format(name)}


def prediction(area_mean:float, concavity_mean:float, concave points_mean:float, radius_worst:float, perimeter_worst:float, area_worst:float, concave points_worst:float, fractal_dimension_worst:float, compactness_worst:float, concave points_se:float, fractal_dimension_mean:float):
    model = joblib.load('Model.pkl')
    make_prediction = model.predict([[area_mean, concavity_mean, concave points_mean, radius_worst, perimeter_worst,area_worst, concave points_worst, fractal_dimension_worst, compactness_worst, concave points_se,fractal_dimension_mean]])
    result = make_prediction[0]
    
    if result == 0:
        return {"This patient is breast cancer free."}
    else:
        return {'This patient is having breast cancer'}


 if __name__ == "__main__":
    uvicorn.run("deployment.app:app", port='127.0.0.1', host='8000')       



