from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess


import xview.models.utils as utils

import matplotlib.pyplot as plt

sys.path.append("models")

from .DeepLabV3_plus import build_deeplabv3_plus


# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

# parser = argparse.ArgumentParser()
# parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs to train for')
# parser.add_argument('--mode', type=str, default="train", help='Select "train", "test", or "predict" mode. \
#     Note that for prediction mode you have to specify an image to run the model on.')
# parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
# parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
# parser.add_argument('--class_balancing', type=str2bool, default=False, help='Whether to use median frequency class weights to balance the classes in the loss')
# parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
# parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
# parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
# parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
# parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
# parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
# parser.add_argument('--num_val_images', type=int, default=10, help='The number of images to used for validations')
# parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
# parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
# parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
# parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
# parser.add_argument('--model', type=str, default="DeepLabV3_plus-Res50", help='The model you are using. Currently supports:\
#     FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, Encoder-Decoder, Encoder-Decoder-Skip, RefineNet-Res50, RefineNet-Res101, RefineNet-Res152, \
#     FRRN-A, FRRN-B, MobileUNet, MobileUNet-Skip, PSPNet-Res50, PSPNet-Res101, PSPNet-Res152, GCN-Res50, GCN-Res101, GCN-Res152, DeepLabV3-Res50 \
#     DeepLabV3-Res101, DeepLabV3-Res152, DeepLabV3_plus-Res50, DeepLabV3_plus-Res101, DeepLabV3_plus-Res152, AdapNet, custom')
# args = parser.parse_args()


args_num_epochs = 2
args_mode = "predict"
args_checkpoint_step = 10
args_validation_step = 1
args_class_balancing = False
#Predict image
args_image = "lindau_000045_000019_leftImg8bit.png"
args_continue_training = False
args_dataset = "CamVid"
args_crop_height = 512
args_crop_width = 512
args_batch_size = 1
args_num_val_images =10
#for data augmentation
args_h_flip = False
args_v_flip = False
args_brightness = None
args_rotation = None
args_model = "DeepLabV3_plus-Res50"

# Get a list of the training, validation, and testing file paths
def prepare_data(dataset_dir=args_dataset):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names


def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image

def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, args_crop_height, args_crop_width)

    if args_h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args_v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args_brightness:
        factor = 1.0 + random.uniform(-1.0*args_brightness, args_brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args_rotation:
        angle = random.uniform(-1*args_rotation, args_rotation)
    if args_rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image

def download_checkpoints(model_name):
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])


def DeepLabV3_plus_main():
    

    print("Dataset -->", args_dataset)
    print("Model -->", args_model)
    print("Crop Height -->", args_crop_height)
    print("Crop Width -->", args_crop_width)
    print("Num Epochs -->", args_num_epochs)
    print("Batch Size -->", args_batch_size)
 

    print("Data Augmentation:")
    print("\tVertical Flip -->", args_v_flip)
    print("\tHorizontal Flip -->", args_h_flip)
    print("\tBrightness Alteration -->", args_brightness)
    print("\tRotation -->", args_rotation)
    # Get the names of the classes so we can record the evaluation results
    class_names_list, label_values = utils.get_label_info(os.path.join(args_dataset, "class_dict.csv"))
    class_names_string = ""
    for class_name in class_names_list:
        if not class_name == class_names_list[-1]:
            class_names_string = class_names_string + class_name + ", "
        else:
            class_names_string = class_names_string + class_name

    num_classes = len(label_values)
    print("Num Classes -->", num_classes)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # config = tf.ConfigProto(device_count={'GPU': 0})
    # sess = tf.Session(config=config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   # True
    sess=tf.Session(config=config)

    # Get the selected model. 
    # Some of them require pre-trained ResNet

    # if "Res50" in args_model and not os.path.isfile("models/resnet_v2_50.ckpt"):
    #     download_checkpoints("Res50")
    # if "Res101" in args_model and not os.path.isfile("models/resnet_v2_101.ckpt"):
    #     download_checkpoints("Res101")
    # if "Res152" in args_model and not os.path.isfile("models/resnet_v2_152.ckpt"):
    #     download_checkpoints("Res152")

    # Compute your softmax cross entropy loss
    print("Preparing the model ...")
    net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 


    network = None
    init_fn = None

    if args_model == "DeepLabV3_plus-Res50" or args_model == "DeepLabV3_plus-Res101" or args_model == "DeepLabV3_plus-Res152":
        # DeepLabV3+ requires pre-trained ResNet weights
        network, init_fn = build_deeplabv3_plus(net_input, preset_model = args_model, num_classes=num_classes)
    else:
        raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")


    losses = None
    if args_class_balancing:
        print("Computing class weights for", args_dataset, "...")
        class_weights = utils.compute_class_weights(labels_dir=args_dataset + "/train_labels", label_values=label_values)
        weights = tf.reduce_sum(class_weights * net_output, axis=-1)
        unweighted_loss = None
        unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
        losses = unweighted_loss * class_weights
    else:
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
    loss = tf.reduce_mean(losses)

    opt = tf.train.AdamOptimizer(0.0001).minimize(loss, var_list=[var for var in tf.trainable_variables()])

    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    utils.count_params()

    # If a pre-trained ResNet is required, load the weights.
    # This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
    if init_fn is not None:
        init_fn(sess)

    # Load a previous checkpoint if desired
    model_checkpoint_name = "checkpoints/latest_model_" + args_model + "_" + args_dataset + ".ckpt"
    #model_checkpoint_name = "/home/ben/test/modular_semantic_segmentation/dataset/exp/861/deeplabv3+/model/model.ckpt"
    #model_checkpoint_name = "/home/ben/test/modular_semantic_segmentation/dataset/exp/861/deeplabv3+/model/model.ckpt-30358"
    #model_checkpoint_name = "/home/ben/test/modular_semantic_segmentation/dataset/exp/861/deeplabv3_mnv2_pascal_train_aug/model.ckpt-10000"
    #model_checkpoint_name = "/home/ben/test/modular_semantic_segmentation/dataset/exp/861/xception_41/model.ckpt"
    #model_checkpoint_name = "/home/ben/test/modular_semantic_segmentation/dataset/exp/861/deeplabv3_cityscapes_train/model.ckpt"

    if args_continue_training or not args_mode == "train":
        print('Loaded latest model checkpoint')
        #saver.restore(sess,tf.train.latest_checkpoint('/home/ben/test/modular_semantic_segmentation/dataset/exp/861/deeplabv3+/model/'))
        saver.restore(sess, model_checkpoint_name)


    avg_scores_per_epoch = []

    # Load the data
    print("Loading the data ...")
    train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = prepare_data()

    if args_mode == "train":

        print("\n***** Begin training *****")
        print("Dataset -->", args_dataset)
        print("Model -->", args_model)
        print("Crop Height -->", args_crop_height)
        print("Crop Width -->", args_crop_width)
        print("Num Epochs -->", args_num_epochs)
        print("Batch Size -->", args_batch_size)
        print("Num Classes -->", num_classes)

        print("Data Augmentation:")
        print("\tVertical Flip -->", args_v_flip)
        print("\tHorizontal Flip -->", args_h_flip)
        print("\tBrightness Alteration -->", args_brightness)
        print("\tRotation -->", args_rotation)
        print("")

        avg_loss_per_epoch = []

        # Which validation images do we want
        val_indices = []
        num_vals = min(args_num_val_images, len(val_input_names))

        # Set random seed to make sure models are validated on the same validation images.
        # So you can compare the results of different models more intuitively.
        random.seed(16)
        val_indices=random.sample(range(0,len(val_input_names)),num_vals)

        # Do the training here
        for epoch in range(0, args_num_epochs):

            current_losses = []

            cnt=0

            # Equivalent to shuffling
            id_list = np.random.permutation(len(train_input_names))

            num_iters = int(np.floor(len(id_list) / args_batch_size))
            st = time.time()
            epoch_st=time.time()
            for i in range(num_iters):
                # st=time.time()
                
                input_image_batch = []
                output_image_batch = [] 

                # Collect a batch of images
                for j in range(args_batch_size):
                    index = i*args_batch_size + j
                    id = id_list[index]
                    input_image = load_image(train_input_names[id])
                    output_image = load_image(train_output_names[id])

                    with tf.device('/cpu:0'):
                        input_image, output_image = data_augmentation(input_image, output_image)


                        # Prep the data. Make sure the labels are in one-hot format
                        input_image = np.float32(input_image) / 255.0
                        output_image = np.float32(utils.one_hot_it(label=output_image, label_values=label_values))
                        
                        input_image_batch.append(np.expand_dims(input_image, axis=0))
                        output_image_batch.append(np.expand_dims(output_image, axis=0))

                # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****
                # input_image = tf.image.crop_to_bounding_box(input_image, offset_height=0, offset_width=0, 
                #                                               target_height=args_crop_height, target_width=args_crop_width).eval(session=sess)
                # output_image = tf.image.crop_to_bounding_box(output_image, offset_height=0, offset_width=0, 
                #                                               target_height=args_crop_height, target_width=args_crop_width).eval(session=sess)
                # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****

                # memory()
                
                if args_batch_size == 1:
                    input_image_batch = input_image_batch[0]
                    output_image_batch = output_image_batch[0]
                else:
                    input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                    output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

                # Do the training
                _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
                current_losses.append(current)
                cnt = cnt + args_batch_size
                if cnt % 20 == 0:
                    string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
                    utils.LOG(string_print)
                    st = time.time()

            mean_loss = np.mean(current_losses)
            avg_loss_per_epoch.append(mean_loss)
            
            # Create directories if needed
            if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
                os.makedirs("%s/%04d"%("checkpoints",epoch))

            # Save latest checkpoint to same file name
            print("Saving latest checkpoint")
            saver.save(sess,model_checkpoint_name)

            if val_indices != 0 and epoch % args_checkpoint_step == 0:
                print("Saving checkpoint for this epoch")
                saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))


            if epoch % args_validation_step == 0:
                print("Performing validation")
                target=open("%s/%04d/val_scores.csv"%("checkpoints",epoch),'w')
                target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))


                scores_list = []
                class_scores_list = []
                precision_list = []
                recall_list = []
                f1_list = []
                iou_list = []


                # Do the validation on a small set of validation images
                for ind in val_indices:
                    
                    input_image = np.expand_dims(np.float32(load_image(val_input_names[ind])[:args_crop_height, :args_crop_width]),axis=0)/255.0
                    gt = load_image(val_output_names[ind])[:args_crop_height, :args_crop_width]
                    gt = utils.reverse_one_hot(utils.one_hot_it(gt, label_values))

                    # st = time.time()

                    output_image = sess.run(network,feed_dict={net_input:input_image})
                    

                    output_image = np.array(output_image[0,:,:,:])
                    output_image = utils.reverse_one_hot(output_image)
                    out_vis_image = utils.colour_code_segmentation(output_image, label_values)

                    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)
                
                    file_name = utils.filepath_to_name(val_input_names[ind])
                    target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
                    for item in class_accuracies:
                        target.write(", %f"%(item))
                    target.write("\n")

                    scores_list.append(accuracy)
                    class_scores_list.append(class_accuracies)
                    precision_list.append(prec)
                    recall_list.append(rec)
                    f1_list.append(f1)
                    iou_list.append(iou)
                    
                    gt = utils.colour_code_segmentation(gt, label_values)
        
                    file_name = os.path.basename(val_input_names[ind])
                    file_name = os.path.splitext(file_name)[0]
                    cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                    cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


                target.close()

                avg_score = np.mean(scores_list)
                class_avg_scores = np.mean(class_scores_list, axis=0)
                avg_scores_per_epoch.append(avg_score)
                avg_precision = np.mean(precision_list)
                avg_recall = np.mean(recall_list)
                avg_f1 = np.mean(f1_list)
                avg_iou = np.mean(iou_list)

                print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
                print("Average per class validation accuracies for epoch # %04d:"% (epoch))
                for index, item in enumerate(class_avg_scores):
                    print("%s = %f" % (class_names_list[index], item))
                print("Validation precision = ", avg_precision)
                print("Validation recall = ", avg_recall)
                print("Validation F1 score = ", avg_f1)
                print("Validation IoU score = ", avg_iou)

            epoch_time=time.time()-epoch_st
            remain_time=epoch_time*(args_num_epochs-1-epoch)
            m, s = divmod(remain_time, 60)
            h, m = divmod(m, 60)
            if s!=0:
                train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
            else:
                train_time="Remaining training time : Training completed.\n"
            utils.LOG(train_time)
            scores_list = []

        fig = plt.figure(figsize=(11,8))
        ax1 = fig.add_subplot(111)

        
        ax1.plot(range(args_num_epochs), avg_scores_per_epoch)
        ax1.set_title("Average validation accuracy vs epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Avg. val. accuracy")


        plt.savefig('accuracy_vs_epochs.png')

        plt.clf()

        ax1 = fig.add_subplot(111)

        
        ax1.plot(range(args_num_epochs), avg_loss_per_epoch)
        ax1.set_title("Average loss vs epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Current loss")

        plt.savefig('loss_vs_epochs.png')

    elif args_mode == "test":
        print("\n***** Begin testing *****")
        print("Dataset -->", args_dataset)
        print("Model -->", args_model)
        print("Crop Height -->", args_crop_height)
        print("Crop Width -->", args_crop_width)
        print("Num Classes -->", num_classes)
        print("")

        # Create directories if needed
        if not os.path.isdir("%s"%("Val")):
                os.makedirs("%s"%("Val"))

        target=open("%s/val_scores.csv"%("Val"),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []
        run_times_list = []

        # Run testing on ALL test images
        for ind in range(len(val_input_names)):
            sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(val_input_names)))
            sys.stdout.flush()

            input_image = np.expand_dims(np.float32(load_image(val_input_names[ind])[:args_crop_height, :args_crop_width]),axis=0)/255.0
            gt = load_image(val_output_names[ind])[:args_crop_height, :args_crop_width]
            gt = utils.reverse_one_hot(utils.one_hot_it(gt, label_values))

            st = time.time()
            output_image = sess.run(network,feed_dict={net_input:input_image})

            run_times_list.append(time.time()-st)

            output_image = np.array(output_image[0,:,:,:])
            output_image = utils.reverse_one_hot(output_image)
            out_vis_image = utils.colour_code_segmentation(output_image, label_values)

            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)
        
            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)
            
            gt = utils.colour_code_segmentation(gt, label_values)

            cv2.imwrite("%s/%s_pred.png"%("Val", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%s_gt.png"%("Val", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_time = np.mean(run_times_list)
        print("Average test accuracy = ", avg_score)
        print("Average per class test accuracies = \n")
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Average precision = ", avg_precision)
        print("Average recall = ", avg_recall)
        print("Average F1 score = ", avg_f1)
        print("Average mean IoU score = ", avg_iou)
        print("Average run time = ", avg_time)


    elif args_mode == "predict":

        if args_image is None:
            ValueError("You must pass an image path when using prediction mode.")

        print("\n***** Begin prediction *****")
        print("Dataset -->", args_dataset)
        print("Model -->", args_model)
        print("Crop Height -->", args_crop_height)
        print("Crop Width -->", args_crop_width)
        print("Num Classes -->", num_classes)
        print("Image -->", args_image)
        print("")
        
        sys.stdout.write("Testing image " + args_image)
        sys.stdout.flush()

        # to get the right aspect ratio of the output
        loaded_image = load_image(args_image)
        height, width, channels = loaded_image.shape
        resize_height = int(height / (width / args_crop_width))

        resized_image =cv2.resize(loaded_image, (args_crop_width, resize_height))
        input_image = np.expand_dims(np.float32(resized_image[:args_crop_height, :args_crop_width]),axis=0)/255.0

        st = time.time()
        output_image = sess.run(network,feed_dict={net_input:input_image})

        run_time = time.time()-st

        output_image = np.array(output_image[0,:,:,:])
        output_image = utils.reverse_one_hot(output_image)

        # this needs to get generalized
        class_names_list, label_values = utils.get_label_info(os.path.join("CamVid", "class_dict.csv"))

        out_vis_image = utils.colour_code_segmentation(output_image, label_values)
        file_name = utils.filepath_to_name(args_image)
        cv2.imwrite("%s/%s_pred.png"%("Test", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

        
        print("")
        print("Finished!")
        print("Wrote image " + "%s/%s_pred.png"%("Test", file_name))
        return cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)

    else:
        ValueError("Invalid mode selected.")
