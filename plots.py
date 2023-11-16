import seaborn as sns
import matplotlib.pyplot as plt

ViT_params = [ [5.71,	22.05,	22.87,	86.56] for i in range(6)]
ViT_accs = [
    [0.7546,0.8139,	0.7599,	0.851],
	[0.002,	0.004,	0.0039,	0.2844],
	[0.0142,0.2198,	0.19,	0.7754],
	[0.091,	0.4896,	0.4141,	0.7954],
	[0.0892,0.4955,	0.4116,	0.7971],
	[0.2579,0.6133,	0.5385,	0.8087],]

sns.lineplot(x=ViT_params, y=ViT_accs)
plt.savefig('temp.png')
plt.close()