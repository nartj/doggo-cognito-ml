import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {IonicModule} from '@ionic/angular';
import {FormsModule} from '@angular/forms';
import {RouterModule} from '@angular/router';

import {HomePage} from './home.page';
import {ImageCropperModule} from 'ngx-image-cropper';


import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'removeUnderscore' })
export class RemoveUnderscorePipe implements PipeTransform {
    transform(value: any, args?: any): any {
        return value.replace(/_/g, " ");
    }
}


@NgModule({
    imports: [
        CommonModule,
        FormsModule,
        IonicModule,
        ImageCropperModule,
        RouterModule.forChild([
            {
                path: '',
                component: HomePage
            }
        ])
    ],
    declarations: [HomePage, RemoveUnderscorePipe]
})
export class HomePageModule {
}
