import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

preds = pd.DataFrame()
subA = pd.read_csv('submission/rf_fe_20160131_tuned.csv')  # 0.47518
# xgb local 5-CV best
subB = pd.read_csv('submission/xgb_script_best_20160225.csv')  # 0.47337

subC = pd.read_csv('submission/rf_script_best_20160224.csv')  # 0.47583
subD = pd.read_csv('submission/xgb_fe_20160210_tune.csv')  # 0.47475
subE = pd.read_csv('submission/download_submission_script.csv')  # 0.470
subF = pd.read_csv('submission/download_submission.csv')
subG = pd.read_csv('submission/rf_fe_20160217_tuned.csv')  # 0.47597
# 0.46875 local 5-CV
subH = pd.read_csv('submission/rf_fe_tuned_20160222.csv')


preds['id'] = subA.id
preds['PredsSubA'] = subA.relevance
preds['PredsSubB'] = subB.relevance
preds['PredsSubC'] = subC.relevance
preds['PredsSubD'] = subD.relevance
preds['PredsSubE'] = subE.relevance
preds['PredsSubF'] = subF.relevance
preds['PredsSubG'] = subG.relevance
preds['PredsSubH'] = subH.relevance
preds['RanksSubA'] = rankdata(subA.relevance)
preds['RanksSubB'] = rankdata(subB.relevance)
preds['RanksSubC'] = rankdata(subC.relevance)
preds['RanksSubD'] = rankdata(subD.relevance)
preds['RanksSubE'] = rankdata(subE.relevance)
preds['RanksSubF'] = rankdata(subF.relevance)
preds['RanksSubG'] = rankdata(subG.relevance)
preds['RanksSubH'] = rankdata(subH.relevance)
preds['RankAverage'] = preds[
    ['RanksSubA', 'RanksSubB', 'RanksSubC', 'RanksSubD', 'RanksSubE', 'RanksSubF', 'RanksSubG', 'RanksSubH']].mean(1)
# preds['FinalBlend'] = MinMaxScaler().fit_transform(
#    preds['RankAverage'].reshape(-1, 1))

#blend = preds['FinalBlend'].values

pred_final = (preds['PredsSubA'] * 1.0 +
              preds['PredsSubB'] * 2.0 +
              preds['PredsSubC'] * 1.0 +
              preds['PredsSubD'] * 1.0 +
              preds['PredsSubE'] * 1.0 +
              preds['PredsSubF'] * 2.0 +
              preds['PredsSubG'] * 1.0 +
              preds['PredsSubH'] * 1.0) / 10.0

submit_df = pd.DataFrame(
    {"id": subA.id, "relevance": pred_final})
submit_df.to_csv(
    'submission/home_depot_fe_rfxgb_average_20160226.csv', index=False)
# submit_df = pd.DataFrame({"id": subA.id, "relevance": blend})

# submit_df.to_csv(
#     'submission/home_depot_fe_rfxgb_average_20160226.csv', index=False)

print submit_df.head()
