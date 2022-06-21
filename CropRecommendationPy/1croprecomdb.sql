-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: May 06, 2022 at 12:13 PM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `1croprecomdb`
--

-- --------------------------------------------------------

--
-- Table structure for table `admintb`
--

CREATE TABLE `admintb` (
  `UserName` varchar(250) NOT NULL,
  `Password` varchar(250) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admintb`
--

INSERT INTO `admintb` (`UserName`, `Password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `querytb`
--

CREATE TABLE `querytb` (
  `id` bigint(250) NOT NULL auto_increment,
  `UserName` varchar(250) NOT NULL,
  `Nitrogen` varchar(250) NOT NULL,
  `Phosphorus` varchar(250) NOT NULL,
  `Potassium` varchar(250) NOT NULL,
  `Temperature` varchar(250) NOT NULL,
  `Humidity` varchar(250) NOT NULL,
  `PH` varchar(250) NOT NULL,
  `Rainfall` varchar(250) NOT NULL,
  `DResult` varchar(250) NOT NULL,
  `CropInfo` varchar(250) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=11 ;

--
-- Dumping data for table `querytb`
--

INSERT INTO `querytb` (`id`, `UserName`, `Nitrogen`, `Phosphorus`, `Potassium`, `Temperature`, `Humidity`, `PH`, `Rainfall`, `DResult`, `CropInfo`) VALUES
(1, 'san', '90', '90', '99', '00', '99', '99', '990', 'Predict', 'papaya'),
(2, 'san123', '90', '90', '99', '22.65', '55', '8', '100', 'Predict', 'chickpea'),
(3, 'san', '90', '90', '99', '22.65', '55', '6.5', '100', 'Predict', 'chickpea'),
(4, 'san', '90', '90', '99', '22.65', '55', '99', '100', 'Predict', 'chickpea'),
(5, 'san567', '70', '120', '100', '12.6', '45', '6.5', '100', 'Predict', 'chickpea'),
(6, 'san', '90', '120', '50', '12.65', '40', '4.5', '200', 'Predict', 'coconut'),
(7, 'san', '90', '120', '99', '22.65', '55', '99', '100', 'waiting', ''),
(8, 'jayanthi', '90', '90', '100', '22.65', '55', '6.5', '200', 'Predict', 'coffee'),
(9, 'san', '170', '180', '220', '45', '200', '20', '400', 'Predict', 'apple'),
(10, 'jayanthi', '70', '10', '10', '9', '15', '4', '100', 'Predict', 'rice');

-- --------------------------------------------------------

--
-- Table structure for table `regtb`
--

CREATE TABLE `regtb` (
  `Name` varchar(250) NOT NULL,
  `Gender` varchar(250) NOT NULL,
  `Age` varchar(250) NOT NULL,
  `Email` varchar(250) NOT NULL,
  `Mobile` varchar(250) NOT NULL,
  `Address` varchar(250) NOT NULL,
  `UserName` varchar(250) NOT NULL,
  `Password` varchar(250) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `regtb`
--

INSERT INTO `regtb` (`Name`, `Gender`, `Age`, `Email`, `Mobile`, `Address`, `UserName`, `Password`) VALUES
('san', 'male', '20', 'sangeeth5535@gmail.com', '9486365535', 'no 6 trichy', 'san', 'san'),
('sanNew', 'male', '20', 'sangeeth5535@gmail.com', '9486365535', 'no ', 'sanNew', 'sanNew'),
('mani', 'male', '33', 'ishu@gmail.com', '9486365535', 'dgh', 'mani', 'mani'),
('san', 'male', '20', 'san@gmail.com', '9486365535', 'dgh', 'san123', 'san123'),
('san', 'male', '20', 'ishu@gmail.com', '09486365535', 'no', 'san567', 'san567'),
('jayanthi', 'female', '20', 'jayanthi@gmail.com', '9994084245', 'No 16, Samnath Plaza, Madurai Main Road, Melapudhur', 'jayanthi', 'jayanthi');
