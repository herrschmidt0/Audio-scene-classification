

class DimReductor:

	def __init__(self, spec, n_components):
		self.spec = spec
		self.n_components = n_components

	def pca(self):
		
		pca = PCA(n_components=self.n_components)
		output = pca.fit_transform(self.spec.T)

		return output.T

	def tsne(self):
		
		if self.spec.shape[1] > 50:
			pca = PCA(n_components=50)
			output_partial = pca.fit_transform(self.spec.T)

		tsne = TSNE(n_components=self.n_components)
		output = tsne.fit_transform(output_partial)
		return output.T

	def isomap(self):
		...

	def som(self):
		...