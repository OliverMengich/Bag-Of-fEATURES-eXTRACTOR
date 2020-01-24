
% type exampleBagOfFeaturesColorExtractor.m

rootfolder = fullfile('101_ObjectCategories');
 outputfolder = fullfile(rootfolder);
 ImageSets1 = imageSet(fullfile(outputfolder,'laptop'));
 
 trainingSets1 = partition(ImageSets1,0.6,'randomize');
 colorBag = bagOfFeatures(trainingSets1,'CustomExtractor',...
      @exampleBagOfFeaturesColorExtractorFunc,'VocabularySize',10000);
 colorbag = bagOfFeatures(trainingSets);
 function[features,metrics] = exampleBagOfFeaturesColorExtractorFunc(I)
  
  [~,~,P] = size(I);
  
  %  isColorImage = P == 3;
   isColorImage = P == 3;
  if isColorImage
     Ilab = rgb2gray(I); 
     Ilab = imresize(Ilab,[227 227]);
     imshow(Ilab);
     
     [Mr,Nr,~] = size(Ilab);
     colorFeatures = reshape(Ilab,Mr*Nr,[]);
     
     rowNorm = sqrt(sum(colorFeatures.^2,2));
     colorFeatures = bsxfun(@rdivide,colorFeatures,rowNorm + eps);
     
     xnorm = linspace(-0.5,0.5,Nr);
     ynorm = linspace(-0.5,0.5,Mr);
     
     [x,y] = meshgrid(xnorm,ynorm);
     
     features = [colorFeatures y(:) x(:)];
     
     metrics = var(colorFeatures(:,1:3),0,2);
  else
     features = zeros(0,5);
     metrics = zeros(0,1);
  end
  end