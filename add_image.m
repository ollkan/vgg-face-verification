function add_image(file_name, desc)
    %load face models
    net_path = 'data/vgg_face.mat';
    model_path = 'data/face_model.mat';
    faces_mat = load('data/faces.mat');
    faces = faces_mat.faces;
    faceDet = lib.face_detector.dpmCascadeDetector(model_path);
    convNet = lib.face_feats.convNet(net_path);

    %empty face struct
    face_structure = struct(...
        'desc', desc, ...
        'images', []);

    %img features fron CNN
    img = imread(file_name);
    det = faceDet.detect(img);
    crop = lib.face_proc.faceCrop.crop(img,det(1:4,1));
    nn_result = convNet.simpleNN(crop);


    %search existing identity
    faces_size = numel(faces);

    found = false;
    if faces_size > 0
        for i=1:faces_size
            if strcmp(faces{i}.desc, desc)
                found = true;
                images_size = numel(faces{i}.images);
                faces{i}.images{images_size + 1} = nn_result;
            end  
        end
    end

    if ~found
        face_structure.desc = desc;
        face_structure.images{1} = nn_result;
        faces{faces_size + 1} = face_structure;
    end

    save data/faces.mat faces;
end