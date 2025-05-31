# C. Cocuzza, 2023, Holmes Lab, Yale University
# A modification to the visualization tools in https://github.com/ColeLab/ActflowToolbox/blob/master/tools/addNetColors_Seaborn.py
# To make those functions more flexible w.r.t inputs / partition / etc. 

# This function will make a seaborn heatmap (https://seaborn.pydata.org/generated/seaborn.heatmap.html) and 
# properly add cluster color labels across the x and y axes. 

# Intended use: average functional connectivity (FC; typically fMRI data) data (or FC matrix for 1 subject, etc.; i.e., 2D only) 
# that has been parcellated using a standard regional atlas, and those regions have been sorted/ordered into brain networks, 
# which you want to label with specific colors along the x and y axes. 

# Can now also be used for any 2D data in which you want to add cluster-style color labels to either the x or y axes. 
# See input documentations below as well as example usages.

########################################################################
# IMPORTS 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns # see: https://seaborn.pydata.org/

########################################################################
# DEFAULT VARIABLES
# The default is to use Yeo 17-network (Yeo et al., 2011, J Neurophysiol) and Schaefer 400 (Schaefer et al., 2018, Cereb Cort) cortical regions 
# Also see: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation

colorList_Yeo7 = [(0.47058823529411764, 0.07058823529411765, 0.5254901960784314),
                  (0.27450980392156865, 0.5098039215686274, 0.7058823529411765),
                  (0.0, 0.4627450980392157, 0.054901960784313725),
                  (0.7686274509803922, 0.22745098039215686, 0.9803921568627451),
                  (0.8627450980392157, 0.9725490196078431, 0.6431372549019608),
                  (0.9019607843137255, 0.5803921568627451, 0.13333333333333333),
                  (0.803921568627451, 0.24313725490196078, 0.3058823529411765)]

colorList_Yeo17 = [(0.04705882, 0.18823529, 1.        ),
                   (1.        , 1.        , 0.        ),
                   (0.80392157, 0.24313725, 0.30588235),
                   (0.        , 0.        , 0.50980392),
                   (0.90196078, 0.58039216, 0.13333333),
                   (0.52941176, 0.19607843, 0.29019608),
                   (0.46666667, 0.54901961, 0.69019608),
                   (0.78431373, 0.97254902, 0.64313725),
                   (0.47843137, 0.52941176, 0.19607843),
                   (0.76862745, 0.22745098, 0.98039216),
                   (1.        , 0.59607843, 0.83529412),
                   (0.29019608, 0.60784314, 0.23529412),
                   (0.        , 0.4627451 , 0.05490196),
                   (0.2745098 , 0.50980392, 0.70588235),
                   (0.16470588, 0.8       , 0.64313725),
                   (0.47058824, 0.07058824, 0.5254902 ),
                   (1.        , 0.        , 0.        )]

netBoundaries_Schaefer400_7 = [(0,  60,  61),
                             (61, 137,  77),
                             (138, 183,  46),
                             (184, 230,  47),
                             (231, 256,  26),
                             (257, 347,  91),
                             (348, 399,  52)]

netBoundaries_Schaefer400_17 = [(0.,  14.,  14.),
                                (14.,  46.,  32.),
                                (46.,  78.,  32.),
                                (78.,  87.,   9.),
                                (87., 116.,  29.),
                                (116., 140.,  24.),
                                (140., 152.,  12.),
                                (152., 164.,  12.),
                                (164., 179.,  15.),
                                (179., 215.,  36.),
                                (215., 236.,  21.),
                                (236., 257.,  21.),
                                (257., 284.,  27.),
                                (284., 321.,  37.),
                                (321., 356.,  35.),
                                (356., 381.,  25.),
                                (381., 400.,  19.)]

########################################################################
# MAIN FUNCTION 

def heatmap_add_net_colors(fcMatrix,
                           colorList=colorList_Yeo17,
                           netBoundaries=netBoundaries_Schaefer400_17,
                           cbarLabel='FC Estimates',
                           xyLabels='Regions',
                           atlasType=None,
                           figW=8,
                           figH=6,
                           netLabelPercent=0.05,
                           cmapHere='seismic',
                           only_Y=False,
                           only_X=False,
                           yTicks=None,
                           yTickLabels=None, 
                           yAxisLabel=None,
                           xTicks=None,
                           xTickLabels=None,
                           xAxisLabel=None,
                           showFig=True,
                           vMin=None,
                           vMax=None):
    
    """ 
    A function to generate a Seaborn heatmap figure with cluster colors added along x/y axes. 
    Typical usage: fMRI FC matrix parcellated per a regional atlas and partitioned into networks; 
    labels added here = standard network colors. 
    NOTE: the first 3 input variables below are the most important, the remainder just fancify the figure :) 
    
    INPUTS 
        fcMatrix        : A node x node (or region x region, or parcel x parcel) array, typically of FC estimates 
                          (in the Glasser parcellation, this would be 360 x 360, and presumably the 'grand mean' 
                          across subjects and states). NOTE: fcMatrix nodes (x & y axes) should be sorted into their network order.
                          ALTERNATIVE: function was originally intended for FC matrices (square, equivalent x and y axes), but it 
                          can now accept any 2D numpy array.
        colorList       : A nested set of lists of RGB color triplets for each cluster (e.g., brain network). 
                          NOTE: these need to be in range 0-1, so divide by 255 if need be. See the default <colorList_Yeo7>
                          variable above for formatting.
        netBoundaries   : A nested set of lists of network "boundary" information. 3 points of info for each nested list 
                          (i.e., would be 3 columns if in numpy array format; and rows would be each network): 
                          1st = start index for that network, 2nd = end index for that network, 3rd = number of regions in that network. 
                          NOTE: indexing in python style (0 start). See the default <netBoundaries_Schaefer400> above for formatting.
        cbarLabel       : Optional; default = 'FC Estimates'. A string to label the heatmap's colorbar (DV).
        xyLabels        : Optional; default = 'Regions'. A string to label the heatmap's x and y labels. 
                          NOTE: if using only_X or only_Y this will be applied to the *other* axis only (see input variables below).
        atlasType       : Optional; default = None. This is to use some common fMRI-based atlases. Current options include: 
                          'CABNP', 'Yeo7_Schaefer400'. More TBA.
        figW            : Optional; default = 8. Figure width (in standard matplotlib format).
        figH            : Optional; default = 6. Figure height (in standard matplotlib format). NOTE: heatmap is square within the panel.
        netLabelPercent : Optional; default = 0.05. Percent (in decimal formatting) of regions to set label size to. 
                          For example, if there are 360 regions on each x/y axis, 5% (0.05) would make the label sizes 18 regions wide.
        cmapHere        : Optional; default = 'seismic'. A string for the heatmap color palette to use. Another good option is 'bwr'. 
                          Follows matplotlib conventions here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        only_Y          : Optional; default = False; if True, will apply the network color labels to only the y axis.
        only_X          : Optional; default = False; if True, will apply the network color labels to only the x axis.
                          NOTE: only_Y and only_X cannot be used together (TBA in future). If both are set, will just do y axis.
        yTicks          : Optional; default = None. Only use with only_X. Vector of locations (scalars) to put y-axis labels. 
                          If None, the axis labels will be removed.
        yTickLabels     : Optional; default = None. Only use with only_X. Vector of strings to put as y-axis labels (at yTicks locations). 
                          If None, the axis labels will be removed.
        yAxisLabel.     : Optional; default = None. Only use wiht only_X. String for y-axis label.
        xTicks          : Optional; default = None. Only use with only_Y. Vector of locations (scalars) to put x-axis labels. 
                          If None, the axis labels will be removed.
        xTickLabels     : Optional; default = None. Only use with only_Y. Vector of strings to put as x-axis labels (at xTicks locations). 
                          If None, the axis labels will be removed.
        xAxisLabel.     : Optional; default = None. Only use wiht only_Y. String for x-axis label.
        showFig         : Optional; default = True. Will show the figure in your IDE.
        vMin            : Optional; default = None (use minimum inherent to data). If you want to force a minimum to colorbar/heatmap.
        vMax            : Optional; default = None (use maximum inherent to data). If you want to force a maximum to colorbar/heatmap.


    OUTPUT
        fig             : a handle for the generated figure, can be used to save it (see example usage below)

            
    EXAMPLE USAGE FOR SQUARE NETWORK DATA (i.e., equivalently adding color labels to both axes): 
        import heatmap_add_net_colors as heatNets
        fcMatrix = avgFC_Unsorted[sortOrderIxs,:][:,sortOrderIxs] # see fcMatrix notes above
        colorList = [(0.47, 0.07, 0.52),(0.27, 0.50, 0.70),(0.0, 0.46, 0.05),(0.76, 0.22, 0.98),(0.86, 0.97, 0.64),(0.90, 0.58, 0.13),(0.80, 0.24 0.30)]
        netBoundaries = [(0,  60,  61),(61, 137,  77),(138, 183,  46),(184, 230,  47),(231, 256,  26),(257, 347,  91),(348, 399,  52)]
        fig = heatNets.heatmap_add_net_colors(fcMatrix,
                                              colorList=colorList,
                                              netBoundaries=netBoundaries,
                                              cbarLabel='FC Estimates (Pearsons r)',
                                              xyLabels='Cortical Regions',
                                              atlasType=None,
                                              figW=10,
                                              figH=7,
                                              netLabelPercent=0.04,
                                              cmapHere='bwr',
                                              only_Y=False,
                                              only_X=False,
                                              yTicks=False,
                                              yTickLabels=None,
                                              yAxisLabel=None,
                                              xTicks=None,
                                              xTickLabels=None,
                                              xAxisLabel=None)
        # NOTE: in an IDE, if you just call function without setting fig variable (or any variable name) as above, it will produce 2 figures.
        # This is because a fig handle is returned for saving purposes; may be able to improve this in the future.
        
    EXAMPLE USAGE FOR ADDING COLOR LABELS TO Y AXIS ONLY (here: x=conditions, y=regions, DV=activations): 
        # Same as above, but call: 
        fig = heatNets.heatmap_add_net_colors(myDataMatrix,
                                              colorList=colorList,
                                              netBoundaries=netBoundaries,
                                              cbarLabel='Average Activations',
                                              xyLabels='Cortical Regions',
                                              atlasType=None,
                                              figW=10,
                                              figH=7,
                                              netLabelPercent=0.04,
                                              cmapHere='bwr',
                                              only_Y=True,
                                              only_X=False,
                                              yTicks=False,
                                              yTickLabels=None,
                                              yAxisLabel=None,
                                              xTicks=np.arange(0,nParcels,50),
                                              xTickLabels=np.tile('test',np.arange(0,nParcels,50).shape[0]),
                                              xAxisLabel='Conditions') 
        # NOTE: for xTicks and xTicksLabels, you'll want to use something specific to your data (not just "test" evenly spaced)
                                              
    EXAMPLE USAGE FOR ADDING COLOR LABELS TO X AXIS ONLY (here: x=regions, y=participants, DV=explained variance): 
        # Same as above, but call: 
        fig = heatNets.heatmap_add_net_colors(myDataMatrix,
                                              colorList=colorList,
                                              netBoundaries=netBoundaries,
                                              cbarLabel='Explained Variance',
                                              xyLabels='Cortical Regions',
                                              atlasType=None,
                                              figW=10,
                                              figH=7,
                                              netLabelPercent=0.04,
                                              cmapHere='bwr',
                                              only_Y=False,
                                              only_X=True,
                                              yTicks=np.arange(0,nParcels,50),
                                              yTickLabels=np.tile('test',np.arange(0,nParcels,50).shape[0]),
                                              yAxisLabel='Participants',
                                              xTicks=None,
                                              xTickLabels=None,
                                              xAxisLabel=None) 
        # NOTE: for yTicks and yTicksLabels, you'll want to use something specific to your data (not just "test" evenly spaced)
  
    EXAMPLE USAGE, WITH DEFAULTS, PLUS SAVING: 
        import heatmap_add_net_colors as heatNets
        fig = heatNets.heatmap_add_net_colors(fcMatrix)
        figDirectory = '/path/to/your/figure/directory/here/'
        figFileName = figDirectory + 'figureName.png'
        fig.savefig(figFileName, bbox_inches='tight', format='png', dpi=250)
        
    TO-DO LIST:
        1. Allow matplotlib strings for colors in addition to RGB triplets
        2. Add a legend option for network label strings 
        3. Allow option for matplotlib heatmaps (not just seaborn)? (i.e., if not installed)
        4. Relatedly, consider an option or version where it's just labels added to existing plot? 
        5. Allow for flexible syntax reading in <atlasType> input variable
        6. Add flexibility: x and y can have different color labels.
        7. Add more options to atlasType
    """ 
    
    # CAB-NP (Ji et al., 2019, NeuroImage) & MMP (Glasser et al., 2016, Nature) network partition/regional parcellation (respectively) variables (cortical)
    if atlasType=='CABNP':
        colorList = [(0, 0, 1),
                     (0.3922, 0, 1),
                     (0, 1, 1),
                     (0.6, 0, 0.6),
                     (0, 1, 0),
                     (0, 0.6, 0.6),
                     (1, 1, 0),
                     (0.98, 0.24, 0.98),
                     (1, 0, 0),
                     (0.7, 0.35, 0.16),
                     (1, 0.6, 0),
                     (0.25, 0.5, 0)]
        netBoundaries = [(0,5,6),
                         (6,59,54),
                         (60,98,39),
                         (99,154,56),
                         (155,177,23),
                         (178,200,23),
                         (201,250,50),
                         (251,265,15),
                         (266,342,77),
                         (343,349,7),
                         (350,353,4),
                         (354,359,6)]
        
    # Yeo 7-network (Yeo et al., 2011, J Neurophysiol) and Schaefer 400 (Schaefer et al., 2018, Cereb Cort) network partition/regional parcellation (respectively) variables (cortical)
    elif atlasType=='Yeo7_Schaefer400':
        colorList = [(0.47058823529411764, 0.07058823529411765, 0.5254901960784314),
                     (0.27450980392156865, 0.5098039215686274, 0.7058823529411765),
                     (0.0, 0.4627450980392157, 0.054901960784313725),
                     (0.7686274509803922, 0.22745098039215686, 0.9803921568627451),
                     (0.8627450980392157, 0.9725490196078431, 0.6431372549019608),
                     (0.9019607843137255, 0.5803921568627451, 0.13333333333333333),
                     (0.803921568627451, 0.24313725490196078, 0.3058823529411765)]

        netBoundaries = [(0,  60,  61),
                         (61, 137,  77),
                         (138, 183,  46),
                         (184, 230,  47),
                         (231, 256,  26),
                         (257, 347,  91),
                         (348, 399,  52)]
        
    #elif atlasType='Yeo17_Schaefer400': # TBA

    # Get number of regions (parcels/nodes) and networks (clusters)
    [nParcels_X,nParcels_Y] = np.shape(fcMatrix)
    if only_Y:
        nParcels = nParcels_Y
        #nParcels = nParcels_X
    elif only_X: 
        nParcels = nParcels_X
        #nParcels = nParcels_Y
    elif not only_Y and not only_X: 
        nParcels = nParcels_Y
        if nParcels_X != nParcels_Y:
            print(f"Network is not square (dimensions: x = {nParcels_X}, y = {nParcels_Y}), please see documentation for function specs and requirements and re-run.")
        
    [numNets,c] = np.shape(colorList)
    #print(f"nParcels: {nParcels}, nParcels_X: {nParcels_X}, nParcels_Y: {nParcels_Y}")
    
    if nParcels_X==nParcels_Y:
        squareBool = True
    elif nParcels_X!=nParcels_Y:
        squareBool = False 
    
    # Set color label size (i.e., "buffer") as percent of number of regions; default = 5% (e.g., 18 of 360)
    buffSize = int(np.round(nParcels * netLabelPercent))
    
    # Make room in fcMatrix for network colors 
    bottomSize = (buffSize,nParcels)
    
    if only_Y:
        topSize = (nParcels_X+buffSize,buffSize)
    elif only_X:
        topSize = (nParcels_Y+buffSize,buffSize)
    elif not only_Y and not only_X:
        topSize = (nParcels+buffSize,buffSize)
        
    bottomBuff = np.zeros(bottomSize)
    topBuff = np.zeros(topSize)
    bottomBuff = (bottomBuff+1)*0
    topBuff = (topBuff+1)*0 # 0 is used to make "buffer" white before adding color labels; this assumes heatmap is centered (white) at 0
    bottomAdd = np.vstack((fcMatrix,bottomBuff))
    #print(f"topBuff: {topBuff.shape}")
    
    if only_Y:
        #print(f"fcMatrix: {fcMatrix.shape}, topBuff: {topBuff.shape}, nParcels_X: {nParcels_X}")
        #fcMatrixWithBuffer = np.hstack((topBuff[:nParcels,:],fcMatrix)).copy()
        fcMatrixWithBuffer = np.hstack((fcMatrix,topBuff[:nParcels_X,:])).copy()
        
    elif only_X: 
        fcMatrixWithBuffer = np.vstack((fcMatrix,bottomBuff)).copy()
        
    elif not only_Y and not only_X: 
        fcMatrixWithBuffer = np.hstack((bottomAdd,topBuff)).copy()
        np.fill_diagonal(fcMatrixWithBuffer, 0) # Assumes network/correlation structure, so self-connections are 0; can be removed if need be 
    
    # Generate figure 
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(figW,figH))
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)
    if only_Y:
        tickLabels = np.arange(0,nParcels_X,buffSize) # Note that tick labels are inserted then removed later to avoid numbers on axes; this could be altered though
    elif only_X:
        tickLabels = np.arange(0,nParcels_Y,buffSize) # Note that tick labels are inserted then removed later to avoid numbers on axes; this could be altered though
    elif not only_Y and not only_X:
        tickLabels = np.arange(0,nParcels,buffSize) # Note that tick labels are inserted then removed later to avoid numbers on axes; this could be altered though
        
    sbH = sns.heatmap(fcMatrixWithBuffer,
                      center=0,
                      cmap=cmapHere,
                      square=squareBool,
                      cbar_kws={'label':cbarLabel,'pad':0.02,'aspect':30},
                      xticklabels=tickLabels,
                      yticklabels=tickLabels,
                      vmin=vMin,
                      vmax=vMax)
    if only_Y:
        plt.ylabel(xyLabels,fontsize=15,labelpad=5)
        plt.xlabel(xAxisLabel,fontsize=15,labelpad=7)
        
    elif only_X: 
        plt.ylabel(yAxisLabel,fontsize=15,labelpad=7)
        plt.xlabel(xyLabels,fontsize=15,labelpad=5)
        
    elif not only_Y and not only_X: 
        plt.ylabel(xyLabels,fontsize=15,labelpad=5)
        plt.xlabel(xyLabels,fontsize=15,labelpad=5)
        
    plt.subplots_adjust(left=None, 
                        bottom=None, 
                        right=1, 
                        top=1, 
                        wspace=1, 
                        hspace=1)
    
    # Add network colors to the "buffered" axes 
    netList = list(range(numNets))
    for net in netList: 
        thisNet = netBoundaries[net]
        netSize = thisNet[2]
        netStart = thisNet[0]
        rectH = patches.Rectangle((netStart-1,nParcels),
                                  netSize,
                                  buffSize,
                                  linewidth=1,
                                  edgecolor=colorList[net],
                                  facecolor=colorList[net])
        
        rectV = patches.Rectangle((nParcels,netStart-1),
                                  buffSize,
                                  netSize,
                                  linewidth=1,
                                  edgecolor=colorList[net],
                                  facecolor=colorList[net])
        if only_Y:
            ax.add_patch(rectV)
            
        elif only_X:
            ax.add_patch(rectH)
            
        elif not only_Y and not only_X:
            ax.add_patch(rectH)
            ax.add_patch(rectV)
    
    # These 2 lines would add a little white box in the corner of the 2 meeting axes; unsilence if need be:
    #rectWhite = patches.Rectangle((nParcels-1,nParcels-1),buffSize,buffSize,linewidth=1,edgecolor='white',facecolor='white')
    #ax.add_patch(rectWhite)

    # set global params & show image 
    plt.box(0)
    if only_Y:
        if str(type(xTicks))!="<class 'NoneType'>":
            ax.set_yticks(tickLabels)
            ax.set_xticks(xTicks,xTickLabels)
            plt.rc('ytick',labelsize=10)
            plt.rc('xtick',labelsize=10)
            ax.tick_params(axis=u'y', which=u'both',length=0)
            ax.xaxis.set_tick_params(pad=0.1)
            rmBothAxLabels = False
        elif str(type(xTicks))=="<class 'NoneType'>":
            rmBothAxLabels = True 
            
    elif only_X:
        if str(type(yTicks))!="<class 'NoneType'>":
            ax.set_yticks(yTicks,yTickLabels)
            ax.set_xticks(tickLabels)
            plt.rc('ytick',labelsize=10)
            plt.rc('xtick',labelsize=10)
            ax.tick_params(axis=u'x', which=u'both',length=0)
            ax.yaxis.set_tick_params(pad=0.1)
            rmBothAxLabels = False
        elif str(type(yTicks))=="<class 'NoneType'>":
            rmBothAxLabels = True 
        
    elif not only_Y and not only_Y:
        rmBothAxLabels = True
        
    if rmBothAxLabels:
        ax.set_yticks(tickLabels)
        ax.set_xticks(tickLabels)
        plt.rc('ytick',labelsize=10)
        plt.rc('xtick',labelsize=10)
        ax.tick_params(axis=u'both', which=u'both',length=0)        
        
    plt.box(0)
    
    cbarHere = sbH.collections[0].colorbar
    cbarHere.ax.tick_params(size=0)
    cbarHere.set_label(cbarLabel,labelpad=10,fontsize=12)
    
    if showFig:
        plt.show()
    
    #return fig, fcMatrixWithBuffer
    return fig, ax, sbH