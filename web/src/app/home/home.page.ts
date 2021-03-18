import {Component} from '@angular/core';
import {Camera, CameraOptions} from '@ionic-native/camera/ngx';
import {Crop} from '@ionic-native/crop/ngx';
import {ImagePicker} from '@ionic-native/image-picker/ngx';
import {WebView} from '@ionic-native/ionic-webview/ngx';
import {RecognitionService} from '../_services/recognition.service';
import {Platform} from '@ionic/angular';
import {ImageCroppedEvent} from 'ngx-image-cropper';


@Component({
    selector: 'app-home',
    templateUrl: 'home.page.html',
    styleUrls: ['home.page.scss'],
})
export class HomePage {
    options: CameraOptions = {
        quality: 100,
        destinationType: this.camera.DestinationType.FILE_URI,
        encodingType: this.camera.EncodingType.JPEG,
        mediaType: this.camera.MediaType.PICTURE,
        correctOrientation: true
    };

    fileUrl: any = null;
    recogRes: any = null;
    processing = false;
    imageChangedEvent: any = '';
    cropping = false;


    constructor(private imagePicker: ImagePicker,
                private crop: Crop,
                private camera: Camera,
                private webview: WebView,
                private recogSvc: RecognitionService,
                public plt: Platform) {
    }

    takePicture() {
        this.recogRes = null;
        this.camera.getPicture(this.options).then((image) => {
            this.fileUrl = image;
            this.cropPicture();
        });
    }

    uploadPicture() {
        this.recogRes = null;
        this.imagePicker.getPictures({maximumImagesCount: 1, outputType: 0}).then((results) => {
            this.fileUrl = results[0];
            this.cropPicture();
        });
    }

    submitCropped() {
        this.cropping = false;
        this.submitPicture(this.fileUrl, true);
    }

    cropPictureForm(event: any): void {
        this.cropping = true;
        this.recogRes = null;
        this.imageChangedEvent = event;
    }

    imageCropped(event: ImageCroppedEvent) {
        this.fileUrl = event.base64;
    }

    cropPicture() {
        this.crop.crop(this.fileUrl, {quality: 100})
            .then(
                newImage => {
                    this.fileUrl = this.pathForImage(newImage);
                    this.submitPicture(newImage);
                },
                error => console.error('Error cropping image', error)
            );
    }

    submitPicture(image, form = false) {
        this.processing = true;
        this.recogSvc.upload(image, form).then(req => {
            // @ts-ignore
            req.subscribe(res => {
                this.recogRes = res;
                this.processing = false;
            }, error => {
                this.recogRes = null;
                this.processing = false;
            });
        });
    }

    pathForImage(img) {
        if (img === null) {
            return '';
        } else {
            return this.webview.convertFileSrc(img);
        }
    }

    getStyle(dog) {
        return {
            'background-image': 'url(assets/dogs/' + dog + '.jpg)'
        };
    }
}
