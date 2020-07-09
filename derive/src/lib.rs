use proc_macro2::TokenStream;
// use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::{
    parse_macro_input, parse_quote, spanned::Spanned, Data, DataStruct, DeriveInput, Fields,
    GenericParam, Generics, Ident,
};

#[proc_macro_derive(TensorLike)]
pub fn derive_tensor_like(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;
    let generics = add_trait_bounds(input.generics);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let f_to_device_impl = derive_f_to_device_impl(&input.data);
    let f_to_kind_impl = derive_f_to_kind_impl(&input.data);
    let shallow_clone_impl = derive_shallow_clone_impl(&input.data);

    let expanded = quote! {
        impl #impl_generics tch_tensor_like::TensorLike for #name #ty_generics #where_clause {
            fn f_to_device(&self, device: tch::Device) -> Result<Self, tch::TchError> {
                #f_to_device_impl
            }

            fn f_to_kind(&self, kind: tch::Kind) -> Result<Self, tch::TchError> {
                #f_to_kind_impl
            }

            fn shallow_clone(&self) -> Self {
                #shallow_clone_impl
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

fn derive_impl<F>(data: &Data, transform: F) -> TokenStream
where
    F: Fn(&Ident) -> TokenStream,
{
    match data {
        Data::Struct(DataStruct {
            fields: Fields::Named(named_fields),
            ..
        }) => {
            let extract_idents = named_fields.named.iter().enumerate().map(|(index, field)| {
                let field_name = &field.ident;
                let proxy_name = format_ident!("_{}", index);
                quote_spanned! {
                    field.span() =>
                        #field_name: #proxy_name
                }
            });

            let recurse = named_fields.named.iter().enumerate().map(|(index, field)| {
                let field_name = &field.ident;
                let proxy_name = format_ident!("_{}", index);
                let expanded = transform(&proxy_name);
                quote_spanned! {
                    field.span() =>
                        #field_name: #expanded
                }
            });

            quote! {
                {
                    let Self {
                        #(#extract_idents),*
                    } = self;

                    Self {
                        #(#recurse),*
                    }
                }
            }
        }
        Data::Struct(DataStruct {
            fields: Fields::Unnamed(unnamed_fields),
            ..
        }) => {
            let extract_idents = unnamed_fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let proxy_name = format_ident!("_{}", index);
                    quote_spanned! {
                        field.span() =>
                            #proxy_name

                    }
                });

            let recurse = unnamed_fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let ident = format_ident!("_{}", index);
                    let expanded = transform(&ident);
                    quote_spanned! {
                        field.span() =>
                            #expanded
                    }
                });

            quote! {
                {
                    let Self (
                        #(#extract_idents),*
                    ) = self;

                    Self (
                        #(#recurse),*
                    )
                }
            }
        }
        Data::Struct(DataStruct {
            fields: Fields::Unit,
            ..
        }) => {
            quote! {
                Self
            }
        }
        Data::Enum(data) => {
            let recurse_variants = data.variants.iter().map(|variant| {
                let var_name = &variant.ident;

                match &variant.fields {
                    Fields::Named(fields) => {
                        let expanded_fields =
                            fields.named.iter().enumerate().map(|(index, field)| {
                                let field_name = &field.ident;
                                let proxy_name = format_ident!("_{}", index);

                                quote_spanned! {
                                    field.span() =>
                                        #field_name: #proxy_name
                                }
                            });

                        let recurse_fields =
                            fields.named.iter().enumerate().map(|(index, field)| {
                                let field_name = &field.ident;
                                let proxy_name = format_ident!("_{}", index);
                                let expanded = transform(&proxy_name);

                                quote_spanned! {
                                    field.span() =>
                                        #field_name: #expanded
                                }
                            });

                        quote! {
                            Self::#var_name { #(#expanded_fields),* } => {
                                Self::#var_name {
                                    #(#recurse_fields),*
                                }
                            }
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let expanded_fields =
                            fields.unnamed.iter().enumerate().map(|(index, field)| {
                                let proxy_name = format_ident!("_{}", index);
                                quote_spanned! {
                                    field.span() =>
                                        #proxy_name
                                }
                            });

                        let recurse_fields =
                            fields.unnamed.iter().enumerate().map(|(index, field)| {
                                let proxy_name = format_ident!("_{}", index);
                                let expanded = transform(&proxy_name);
                                quote_spanned! {
                                    field.span() =>
                                        #expanded
                                }
                            });

                        quote! {
                            Self::#var_name ( #(#expanded_fields),* ) => {
                                Self::#var_name (
                                    #(#recurse_fields),*
                                )
                            }
                        }
                    }
                    Fields::Unit => {
                        quote! {
                            Self::#var_name => Self::#var_name
                        }
                    }
                }
            });

            quote! {
                match self {
                    #(#recurse_variants)*
                }
            }
        }
        Data::Union(_data) => panic!("Union type is not supported"),
    }
}

fn derive_f_to_device_impl(data: &Data) -> TokenStream {
    let expanded = derive_impl(data, |ident| {
        quote_spanned! {
            ident.span() =>
                #ident.f_to_device(device)?
        }
    });

    quote! {
        Ok(#expanded)
    }
}

fn derive_f_to_kind_impl(data: &Data) -> TokenStream {
    let expanded = derive_impl(data, |ident| {
        quote_spanned! {
            ident.span() =>
                #ident.f_to_kind(kind)?
        }
    });

    quote!{
        Ok(#expanded)
    }
}

fn derive_shallow_clone_impl(data: &Data) -> TokenStream {
    let expanded = derive_impl(data, |ident| {
        quote_spanned! {
            ident.span() =>
                #ident.shallow_clone()
        }
    });

    expanded
}

fn add_trait_bounds(mut generics: Generics) -> Generics {
    for param in &mut generics.params {
        if let GenericParam::Type(type_param) = param {
            type_param
                .bounds
                .push(parse_quote!(tch_tensor_like::TensorLike));
        }
    }
    generics
}
