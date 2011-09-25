function img = pca_face_reconstruct(index, pca_face_db)

% Reconstructs a training image from a face recognition database.
%
% Arguments:
%   index:        a scalar with the index of the requested training image.
%   pca_face_db:  a string with the path of the file where the database will
%                 be stored.
%
% Returns:
%   A column vector with the pixels of the reconstructed training image.
%
% (c) Jaime Soto
% CAP 6411 - Computer Vision Systems
% University of Central Florida
% 7 December 2010
%

    load(pca_face_db);
    c = components(:,index);
    img = eigenfaces*c + mean_img;
end
