clc;
clear all;
%% Video RobustPCA example: separates background and foreground
addpath('../');

% The movie will be downloaded from the internet
movieFile = 'RobustPCA_video_demo.avi';
urlwrite('http://dlaptev.org/other/RobustPCA_video_demo.avi', movieFile);

% Opening the movie and setting the parameter values.
n_frames = 180;
movie = VideoReader(movieFile);
frate = movie.FrameRate;    
height = movie.Height;
width = movie.Width;

% Vectorizing every frame to form the data matrix X 
X = zeros(n_frames, height*width);
for i = (1:n_frames)
    frame = read(movie, i);
    frame = rgb2gray(frame);
    X(i,:) = reshape(frame,[],1);
end

% Applying Robust PCA
lambda = 1/sqrt(max(size(X)));
tic
[L,S] = RobustPCA(X, lambda/3, 10*lambda/3, 1e-7);
toc

% Preparing the new movie file.
vidObj = VideoWriter('RobustPCA_video_output.avi');
vidObj.FrameRate = frate;
open(vidObj);
range = 255;
map = repmat((0:range)'./range, 1, 3);
S = medfilt2(S, [5,1]);                         % median filter in time
for i = (1:size(X, 1));
    v = X(i,:);
    frame1 = reshape(X(i,:),height,[]);         % Reshaping the X, L and S matrices.
    frame2 = reshape(L(i,:),height,[]);
    frame3 = reshape(abs(S(i,:)),height,[]);
    % median filter in space; threshold
    frame3 = (medfilt2(abs(frame3), [5,5]) > 5).*frame1;
    frame = mat2gray([frame1, frame2, frame3]); % Stacking X, L and S matrices together.
    frame = gray2ind(frame,range);
    frame = im2frame(frame,map);
    writeVideo(vidObj,frame);                       
end
close(vidObj);
