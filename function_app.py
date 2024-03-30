import azure.functions as func
import logging
import json
import xgboost as xgb
import pandas as pd

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
   
@app.route(route="data")
def data(req: func.HttpRequest) -> func.HttpResponse:
    data = {'data': [0,1,2,3,4,5,6,7,8,9]}
    return func.HttpResponse(
        json.dumps(data), 
        mimetype='text/json'
    )  

@app.route(route="project")
def project(req: func.HttpRequest) -> func.HttpResponse:
    data = req.get_json()
    model = xgb.XGBRegressor()
    model.load_model('./model.json')
    tmp = { 'region':[data['region']],\
            'BG.GSR.NFSV.GD.ZS':[data['gdp']],\
            'BM.GSR.CMCP.ZS':[data['computer']],\
            'SE.PRM.ENRL.TC.ZS':[data['student']],\
            'SE.SEC.CUAT.LO.ZS':[data['secondary']],\
            'SH.H2O.BASW.ZS':[data['water']],\
            'SH.STA.BASS.ZS':[data['sanitation']]}
    df = pd.DataFrame(tmp)
    df['region'] = df['region'].astype('category')
    prediction = model.predict(df.iloc[:1])
    return func.HttpResponse(
        json.dumps(float(prediction[0])), 
        mimetype='text/json'
    )