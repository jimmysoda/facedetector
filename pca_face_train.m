function pca_face_train(directory, pattern, tsize, pca_face_db, nimages = -1, ncomponents = 30)

% Creates a face recognition/detection database from a set of the training images.
% All the training images must have the same size.
%
% Arguments:
%   directory:    a string with the path to the directory that contains the 
%                 training images.
%   pattern:      a string with the file pattern that matches the training image 
%                 file names, e.g. '*.pgm'.
%   tsize:        a two-element vector with the size of the training images, 
%                 e.g. [19 19] for 19 x 19 px.
%   pca_face_db:  a string with the path of the file where the database will
%                 be stored.
%   nimages:      a scalar with the number of training images to consider. Reads
%                 all files that match @pattern in @directory by default.          
%   ncomponents:  the number of components (eigenfaces) to include in the database.
%
% (c) Jaime Soto
% CAP 6411 - Computer Vision Systems
% University of Central Florida
% 7 December 2010
%

    workingdir = pwd();
    cd(directory);
    filelist = dir(pattern);
    numrows = tsize(1);
    numcols = tsize(2);

    if (nimages == -1)
	    nimages = length(filelist)
    else
	    nimages = min(nimages, length(filelist));
    end

    images = zeros(numrows*numcols, nimages);

    % For each image in directory
    for f = 1:nimages
	    image = imread(filelist(f).name);

        if (rows(image) ~= numrows || columns(image) ~=numcols)
            fprintf(1, 'Invalid image dimensions: %s (%d, %d)\n', ...
            		filelist(f).name, rows(image), columns(image));
            continue;
        end
	
	    if (isrgb(image))
		    image = rgb2gray(image);
	    end

	    %images(:,f) = double(image(:));

        % Optional histogram equalization - commented out for speedup
	    % Unfold image into a single column
	    % Scale values to (0,1)
	    % Perform histogram equalization
	    % Rescale to (0, 255)	
	    images(:,f) = histeq(double(image(:))./255.0).*255.0;
    end

    cd(workingdir);
        	
    % Compute mean image
    mean_img = mean(images')';

    % Subtract mean from each image
    images = images - repmat(mean_img, [1 nimages]);

    % Obtain eigenvectors and eigenvalues
    % Use method proposed by Turk and Pentland (1991)
    [eigenvectors eigenvalues] = eig(images' * images);

    % Pick the eigenvectors that correspond to the largest eigenvalues
    eigenvectors = eigenvectors(:, 1:ncomponents);

    % Calculate image components
    eigenfaces = images * eigenvectors;
    components = zeros(ncomponents,nimages);

    for i = 1:nimages
	    components(:,i) = eigenfaces' * images(:,i);
    end

    save(pca_face_db, 'mean_img', 'eigenfaces', 'components');
end
