from lime import lime_image
import numpy as np
from tqdm import tqdm

class SquareGrid:

    def __init__(self):
        
        self.model = None
        self.input_size = None
        self.size_basegrid = None
        self.sizes = []
        self.size = None
        self.pow2 = None
        self.heatmaps = []
        self.top_indexes = None
   
    def segment_squaregrid(self,image):

        temp = []
        current_row = []
        new_row = []

        size = self.size
        size_basegrid  = self.size_basegrid
        
        base_square = np.ones((size,size))
        reps = int(size_basegrid/size)

        temp = np.concatenate((base_square,2*base_square),axis=1)

        for num in range(2,reps):
            temp = np.concatenate((temp,(1+num)*base_square),axis=1)

        current_row = temp

        for num in range(1,reps):
            
            new_row = current_row + reps*np.ones(current_row.shape)
            temp = np.concatenate((temp,new_row))
            current_row = new_row

        final = temp - np.ones(temp.shape)    
            
        return final.astype(int)


    def segment_squaregrid_general(self,ima):
        
        size_basegrid = self.size_basegrid

        #Create a square array which is larger than or equal to the largest dimension of our image.
        #It's dimensions, size_basegrid, are the first power of 2 to surpass the largest dimensin of the image
        #We do this because our squaregrids will be all the lower powers of 2 that divide this base grid.
        basegrid = np.zeros((size_basegrid,size_basegrid,3))
        #Segment this base square using a grid of size self.size
        basegrid = self.segment_squaregrid(basegrid)
        #Compute the difference between the size of the base grid and the image, for each dimension
        #These differences are divided in two, to aid in centralizing the base_grid and cropping out a grid the size of 
        #our image.
        diff_dim0 = int(np.floor((basegrid.shape[0]-ima.shape[0])/2))
        diff_dim1 = int(np.floor((basegrid.shape[1]-ima.shape[1])/2))
        #Crop an array the size of the image from base_grid, centralized.
        final = basegrid[diff_dim0:diff_dim0+ima.shape[0],diff_dim1:diff_dim1+ima.shape[1]]
        final = (final - final.min()*np.ones(final.shape)).astype('int')
        
        #After cropping the numbers of the segments might not make sense, since we might have cropped out
        #segments 1,2,3... so we renumber this final grid.
        renumberdict = dict(zip(list(np.unique(final)),range(len(list(np.unique(final))))))
        
        final = np.vectorize(renumberdict.get)(final)
        
        return final.astype(int)

    def explain(self,
                image,
                model,
                hide_color=0, 
                num_samples=1000, 
                batch_size=200,
                top_labels = 1, 
                progress_bar = False,
                verbose = 0,
                min_size = 8
                ):

        #Helper variable to calculate the size of basegrid.
        pow2 = int(np.ceil(np.log(np.max((image.shape[0],image.shape[1])))/np.log(2)))
        self.pow2 = pow2
        self.size_basegrid = 2**pow2
        self.sizes = [2**i for i in range(3,pow2-1) if 2**i>=min_size]
        self.model = model
        
        #heatmaps = []
        explanations_all_sizes = []
        if(verbose):
            print('Generating explanations...')
            print('Batch size is: ', batch_size)
            print('Number of samples is: ', num_samples)
        for size in self.sizes:
            
            if(verbose):
                print('Generating explanation for grid with squares sized ', size)

            self.size = size
            segments = self.segment_squaregrid_general(image)
            
            explainer_sqgrid = lime_image.LimeImageExplainer()
            explanation_squaregrid = explainer_sqgrid.explain_instance(image, 
                                                                       model.predict, 
                                                                       segmentation_fn = self.segment_squaregrid_general, 
                                                                       top_labels=top_labels, 
                                                                       hide_color=hide_color, 
                                                                       num_samples=num_samples, 
                                                                       batch_size=batch_size, 
                                                                       progress_bar = progress_bar 
                                                                       )
            explanations_all_sizes.append(explanation_squaregrid)

        if(verbose):
            print('Generating heatmaps for top',top_labels,'classes',flush=True)
            
        

        predis = model.predict(np.expand_dims(image,0))
        top_indexes = np.flip(np.argsort(predis).squeeze()).tolist()

        heatmaps = []
        for ind in tqdm(top_indexes[:top_labels]) if verbose else top_indexes[:top_labels]:
            heatmaps_temp = []
            for item in explanations_all_sizes:
                dict_squaregrid = dict(item.local_exp[ind])
                cm_squaregrid = np.vectorize(dict_squaregrid.get)(item.segments)
                heatmaps_temp.append(cm_squaregrid)
            final_heatmap = np.sum(heatmaps_temp[:],axis=0)
            heatmaps.append(final_heatmap)

        self.heatmaps = heatmaps
        self.top_indexes = top_indexes
        heatmaps = dict(zip(top_indexes,heatmaps))
        return heatmaps
