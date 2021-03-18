import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {File} from '@ionic-native/file/ngx';

@Injectable({
    providedIn: 'root'
})
export class RecognitionService {

    constructor(private http: HttpClient, private file: File) {
    }

    upload(image, isForm) {
        if (isForm) {
            return new Promise(resolve => resolve(this.send(image)));
        } else {
            const path = image.match(/(.*\/)(.*)(\?.*)/);
            return this.file.readAsDataURL(path[1], path[2]).then(img => {
                return this.send(img);
            });
        }
    }

    private send(img) {
        return this.http.post('http://restartpack.co.uk:8000/upload', {image: img});
    }
}

