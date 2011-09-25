function [difs, dffs] = pca_face_detect(directory, pattern, pca_face_db, nimages = -1)

% Performs face detection for a set of test images using a face recognition database.
% All the training images must have the same size as those used in the training
% stage of the face recognition database.
%
% Arguments:
%   directory:    a string with the path to the directory that contains the 
%                 test images.
%   pattern:      a string with the file pattern that matches the test image 
%                 file names, e.g. '*.pgm'.
%   pca_face_db:  a string with the path of the PCA face database.
%   nimages:      a scalar with the number of test images to consider. Reads
%                 all files that match @pattern in @directory by default.          
% Returns:
%   difs:         a column vector with the distances in face space in log scale.
%   dffs:         a column vector with the distances from face space in log scale.
%
% (c) Jaime Soto
% CAP 6411 - Computer Vision Systems
% University of Central Florida
% 7 December 2010
%
    load(pca_face_db);
    workingdir = pwd();
    cd(directory);
    filelist = dir(pattern);

    if (nimages == -1)
	    nimages = length(filelist)
    else
	    nimages = min(nimages, length(filelist));
    end
    
    difs = zeros(nimages, 1);
    dffs = zeros(nimages, 1);
    face = zeros(nimages, 1);

    for f = 1:nimages
	    image = imread(filelist(f).name);

        if (length(image(:)) ~= length(mean_img))
            continue;
        end
	
	    if (isrgb(image))
		    image = rgb2gray(image);
	    end
	
	    image = double(image);
	    image = histeq(image./255.0).*255.0;
        
        % Project image to face space
	    c = eigenfaces' * (image(:) - mean_img);
	    
	    % Reconstruct the image
	    r = eigenfaces*c + mean_img;
	    
	    % Compute the distance from face space and distance in face space
	    % (Moghaddam and Pentland 1997) 
	    dffs(f) = log(sum(double(image(:).^2) - r.^2));
	    difs(f) = log(mahalanobis(c', components'));
    end

    cd(workingdir);
end
