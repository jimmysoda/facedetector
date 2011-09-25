function pca_face_demo(traindir, facedir, nonfacedir)

% Demonstrates the PCA face recognition/detection algorithm.
%
% Arguments:
%   traindir:     a string with the path to the directory that contains the 
%                 training images.
%
%   facedir:      a string with the path to the directory that contains the 
%                 face test images.
%
%   nonfacedir:   a string with the path to the directory that contains the 
%                 non-face test images.
%
% (c) Jaime Soto
% CAP 6411 - Computer Vision Systems
% University of Central Florida
% 7 December 2010
%   
    pca_face_db = 'pca_face_db.mat';
	tic();
	pca_face_train(traindir, 'face0*.pgm', [19 19], pca_face_db, 200, 200);
	fprintf(1, 'training took %f sec\n', toc());
	
	tic();
	[fdifs fdffs] = pca_face_detect(facedir, 'cmu_0*.pgm', pca_face_db, 1000);
	fprintf(1, 'detection on %d faces took %f sec\n', length(fdifs), toc());
	
	tic();
	[ndifs ndffs] = pca_face_detect(nonfacedir, 'cmu_0*.pgm', pca_face_db, 1000);
	fprintf(1, 'detection on %d non-faces took %f sec\n', length(ndifs), toc());
	
	threshold = 7.75;
	face = fdifs < threshold;
	nface = ndifs < threshold;

	fprintf(1, 'True positives: %d\n', nnz(face));
	fprintf(1, 'True negatives: %d\n', nnz(nface));
	fprintf(1, 'False positives: %d\n', nnz(~nface));
	fprintf(1, 'False negatives: %d\n', nnz(~face));
	clf
	hold on
	title('Feature Space Distances for Faces and Non-Faces');
	plot(fdifs, fdffs, 'ob');
	plot(ndifs, ndffs, 'xr');
	legend('Faces', 'Non-Faces');
	xlabel('[log] Distance in face space (DIFS): Mahalanobis difference');
	ylabel('[log] Distance from face space (DFFS): Sum of Difference of Squares');
	print('facespace_distances.png', '-dpng');
	hold off
end
