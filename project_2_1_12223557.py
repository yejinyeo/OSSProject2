# import module
import pandas as pd
import numpy as np

# load csv file
df = pd.read_csv("2019_kbo_for_kaggle_v2.csv")


print("-"*50)   # 구현 구분선
# implement 1
for season in range(2015, 2019):
  df_season = df[df.year == season]
  df_season_H = df_season[['batter_name', 'H']]
  df_season_avg = df_season[['batter_name', 'avg']]
  df_season_HR = df_season[['batter_name', 'HR']]
  df_season_OBP = df_season[['batter_name', 'OBP']]

  H_top_10 = df_season_H.sort_values(by='H',ascending=False).head(10).batter_name.values
  avg_top_10 = df_season_avg.sort_values(by='avg',ascending=False).head(10).batter_name.values
  HR_top_10 = df_season_HR.sort_values(by='HR',ascending=False).head(10).batter_name.values
  OBP_top_10 = df_season_OBP.sort_values(by='OBP',ascending=False).head(10).batter_name.values

  season_top_10_dic = {'안타(H)':H_top_10, '타율(avg)':avg_top_10, '홈런(HR)':HR_top_10, '출루율(OBP)':OBP_top_10}
  season_top10_df = pd.DataFrame(season_top_10_dic, index=np.arange(1,11))
  season_top10_df.index.name = season

  print(season_top10_df)
  print()


print("-"*50)   # 구현 구분선
# implement 2
position_info = np.unique(df.cp.values)
names = []
df_2018 = df[df.year == 2018]

for position in position_info:
  top_1_name = df_2018[df_2018.cp.values == position][['batter_name', 'war']].sort_values(by='war',ascending=False).batter_name.values[0]
  names.append(top_1_name)

war_players = pd.Series(names, index=position_info)
print(war_players)
print()


print("-"*50)   # 구현 구분선
# implement3
areas = np.array(['R', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG'])
corrs = []

for area in areas:
  corr_salary = df[area].corr(df.salary)
  corrs.append(corr_salary)

computed_corrs = pd.Series(corrs, index=areas)
hig_corr_area = areas[np.array(corrs).argmax()]

print("Result of correlation:")
print(computed_corrs)
print(f"The highest correlation with salary: {hig_corr_area}")