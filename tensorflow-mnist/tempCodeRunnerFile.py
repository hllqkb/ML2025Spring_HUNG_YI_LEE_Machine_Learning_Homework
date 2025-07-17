
# (x_train,y_train),(x_test,y_test)=mnist.load_data()
# x_train,x_test=x_train/255.0,x_test/255.0
# # imgs,labs=x_test[:10],y_test[:10]
# # plot_images=np.hstack(imgs)
# # plt.imshow(plot_images,cmap='gray')
# # plt.show()
# # add a channel dimension to the images
# x_train=x_train[...,tf.newaxis]
# x_test=x_test[...,tf.newaxis]
# # create data generator
# train_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
# test_ds=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)
# model=MyModel()

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_ds,epochs=5,validation_data=test_ds)

# model.evaluate(test_ds)
