import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("literature.csv")
df['colors'] = 'black'
df['scatter'] = None

df.loc[df.method == 'aircraft', 'colors'] = 'darkgreen'
df.loc[df.method == 'aircraft + model', 'colors'] = 'blue'
df.loc[df.method == 'satellite', 'colors'] = 'darkred'
# df.loc[df.kind != 'NO', 'scatter'] = None
df.loc[df.kind == 'NO', 'scatter'] = df.loc[df.kind == 'NO', 'colors']

df.sort_values(['kind','year'], inplace=True, ascending=[False, True])
df = df.reset_index(drop=True)

# plot lines
plt.figure(figsize=(12, 10))
plt.hlines(y=df.index, xmin=df['min'], xmax=df['max'],
           color=df.colors, alpha=0.4, linewidth=1)

# set kins of scatter
x_a = df.loc[df.method == 'aircraft', 'min']
x_ac = df.loc[df.method == 'aircraft + model', 'min']
x_s = df.loc[df.method == 'satellite', 'min']

# plot min scatters
plt.scatter(x_a, x_a.index, alpha=0.6,
            color=df.colors[x_a.index], label='Aircraft')
plt.scatter(x_ac, x_ac.index, alpha=0.6,
            color=df.colors[x_ac.index], label='Aircraft + Model')
plt.scatter(x_s, x_s.index, alpha=0.6,
            color=df.colors[x_s.index], label='Satellite')

# plt max scatters
plt.scatter(df['max'], df.index, color=df.colors, alpha=0.6)

# plot ticks
plt.yticks(df.index, df.agg('{0[author]} et al. ({0[year]})'.format, axis=1))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Production Efficieny / Flash', fontsize=15)

# Add Patches
NO = patches.Rectangle((40, 8.5), width=680, height=4,
                       alpha=0.1, color='grey', lw=1, ls='--', label='NO')
NOx = patches.Rectangle((1, -0.5), width=500, height=9,
                        alpha=0.1, color='red', lw=1, ls='--', label='NOx')
# NO = patches.Rectangle((40, 8.5), width=680, height=4,
#                        alpha=1, fill=False, ec='grey', lw=1, ls='--', label='NO')
# NOx = patches.Rectangle((1, -0.5), width=500, height=9,
#                        alpha=1, fill=False, ec='red', lw=1, ls='--', label='NO')
plt.gca().add_patch(NO)
plt.gca().add_patch(NOx)

# Decorate
plt.title('Literature Estimates', fontdict={'size': 15})
plt.legend(handles=[NO, NOx])
plt.legend(prop={'size': 15})
plt.grid(linestyle='--', alpha=0.5)
plt.savefig('literature_estimates.png', bbox_inches='tight')