CREATE DATABASE accounts;

CREATE TABLE `accounts`.newusers` 
(
    `id` INT NOT NULL AUTO_INCREMENT,
    `username` VARCHAR(100) NOT NULL,
    `email` VARCHAR(100) NOT NULL,
    `password` VARCHAR(100) NOT NULL,
    `avatar` VARCHAR(100) NOT NULL,
PRIMARY KEY (`username`) 
);